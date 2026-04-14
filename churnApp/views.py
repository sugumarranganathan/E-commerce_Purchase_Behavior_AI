import os
import pickle
import pandas as pd

from django.conf import settings
from django.shortcuts import render
from django.views import View

from .forms import ChurnForm


class PredictView(View):
    template_name = 'create.html'

    def get(self, request):
        form = ChurnForm()
        return render(request, self.template_name, {'form': form})

    def post(self, request):
        form = ChurnForm(request.POST)

        if form.is_valid():
            data = form.cleaned_data

            # Create input DataFrame in the same feature order used during training
            input_df = pd.DataFrame([{
                "Gender": data["Gender"],
                "Age": data["Age"],
                "MembershipType": data["MembershipType"],
                "PreferredCategory": data["PreferredCategory"],
                "TotalOrders": data["TotalOrders"],
                "TotalSpent": data["TotalSpent"],
                "LastPurchaseDaysAgo": data["LastPurchaseDaysAgo"],
                "PurchaseFrequencyPerMonth": data["PurchaseFrequencyPerMonth"],
                "AppLoginFrequency": data["AppLoginFrequency"],
                "CouponUsageCount": data["CouponUsageCount"],
                "ReturnCount": data["ReturnCount"],
                "SatisfactionScore": data["SatisfactionScore"],
            }])

            # File paths
            encoders_path = os.path.join(settings.BASE_DIR, "fashion_rf_label_encoders.pkl")
            target_encoder_path = os.path.join(settings.BASE_DIR, "fashion_rf_target_encoder.pkl")
            model_path = os.path.join(settings.BASE_DIR, "fashion_churn_rf_model.pkl")

            # Load encoders and model safely
            with open(encoders_path, "rb") as f:
                label_encoders = pickle.load(f)

            with open(target_encoder_path, "rb") as f:
                target_encoder = pickle.load(f)

            with open(model_path, "rb") as f:
                model = pickle.load(f)

            # Encode categorical columns
            for col in input_df.columns:
                if col in label_encoders:
                    try:
                        input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
                    except ValueError:
                        # Fallback if an unexpected category appears
                        input_df[col] = 0

            # Predict
            pred = model.predict(input_df)
            final_result = target_encoder.inverse_transform(pred)[0]

            # User-friendly output messages + dynamic type
            if str(final_result).strip().lower() == "yes":
                result = "Customer May Not Continue Purchasing"
                prediction_type = "risk"
            else:
                result = "Customer Likely to Continue Purchasing"
                prediction_type = "safe"

            # Confidence score (if model supports predict_proba)
            confidence = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)[0]
                confidence = round(max(probs) * 100, 2)

            return render(request, 'result.html', {
                'result': result,
                'raw_result': final_result,
                'confidence': confidence,
                'prediction_type': prediction_type
            })

        return render(request, self.template_name, {'form': form})
