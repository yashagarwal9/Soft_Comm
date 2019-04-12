from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.index),
    url(r'^process$', views.create_survey),
    url(r'^result$', views.submitted_info),
]
