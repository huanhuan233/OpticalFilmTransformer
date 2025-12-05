from django.urls import path
from .views import infer, calculate_spectrum, optimize_structure

urlpatterns = [
    path('infer/', infer, name='infer'),  # ✅ /api/optogpt/infer/
    path('calculate-spectrum/', calculate_spectrum, name='calculate_spectrum'),  # ✅ /api/optogpt/calculate-spectrum/
    path('optimize-structure/', optimize_structure, name='optimize_structure'),  # ✅ /api/optogpt/optimize-structure/
]
