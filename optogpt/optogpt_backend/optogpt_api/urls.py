from django.urls import path
from .views import infer

urlpatterns = [
    path('infer/', infer, name='infer'),  # ✅ /api/optogpt/infer/
]
