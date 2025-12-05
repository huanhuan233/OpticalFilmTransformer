from django.urls import path
from .views import infer, calculate_spectrum

urlpatterns = [
    path('infer/', infer, name='infer'),  # ✅ /api/optogpt/infer/
    path('calculate-spectrum/', calculate_spectrum, name='calculate_spectrum'),  # ✅ /api/optogpt/calculate-spectrum/
]
