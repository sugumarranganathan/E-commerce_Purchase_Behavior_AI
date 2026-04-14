from django import forms
from .models import ChurnModel


class ChurnForm(forms.ModelForm):
    GENDER_CHOICES = [
        ('M', 'Male'),
        ('F', 'Female'),
    ]

    MEMBERSHIP_CHOICES = [
        ('Free', 'Free'),
        ('Silver', 'Silver'),
        ('Gold', 'Gold'),
        ('Platinum', 'Platinum'),
    ]

    CATEGORY_CHOICES = [
        ('Men', 'Men'),
        ('Women', 'Women'),
        ('Kids', 'Kids'),
        ('Beauty', 'Beauty'),
    ]

    Gender = forms.ChoiceField(
        choices=GENDER_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    MembershipType = forms.ChoiceField(
        choices=MEMBERSHIP_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    PreferredCategory = forms.ChoiceField(
        choices=CATEGORY_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    class Meta:
        model = ChurnModel
        fields = '__all__'
        widgets = {
            'Age': forms.NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter age'
            }),
            'TotalOrders': forms.NumberInput(attrs={
                'class': 'form-control'
            }),
            'TotalSpent': forms.NumberInput(attrs={
                'class': 'form-control',
                'step': 'any'
            }),
            'LastPurchaseDaysAgo': forms.NumberInput(attrs={
                'class': 'form-control'
            }),
            'PurchaseFrequencyPerMonth': forms.NumberInput(attrs={
                'class': 'form-control',
                'step': 'any'
            }),
            'AppLoginFrequency': forms.NumberInput(attrs={
                'class': 'form-control'
            }),
            'CouponUsageCount': forms.NumberInput(attrs={
                'class': 'form-control'
            }),
            'ReturnCount': forms.NumberInput(attrs={
                'class': 'form-control'
            }),
            'SatisfactionScore': forms.NumberInput(attrs={
                'class': 'form-control',
                'step': 'any'
            }),
        }
