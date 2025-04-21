from django import forms
from .models import TradeBlog

class BlogForm(forms.ModelForm):
    class Meta:
        model = TradeBlog
        fields = ['title', 'content', 'chart_image']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'form-control form-control-lg rounded-3',
                'placeholder': 'Enter a catchy title...'
            }),
            'content': forms.Textarea(attrs={
                'class': 'form-control rounded-3',
                'rows': 6,
                'placeholder': 'Write your analysis here...'
            }),
            'chart_image': forms.ClearableFileInput(attrs={
                'class': 'form-control'
            }),
        }
        