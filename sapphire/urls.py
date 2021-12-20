"""sapphire URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from sapphire import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('', views.home, name="home"),
    path('target-variable/', views.tvar, name="tvar"),
    path('null-column/', views.column, name="column"),
    path('outlier/', views.outlier, name="outlier"),
    path('train-test-split/', views.split, name="split"),
    path('final-predictions/', views.predictions, name="predictions"),
    path('about/', views.about, name="about"),
    path('reset/', views.reset, name="reset"),
    path('admin/', admin.site.urls),
    path('sample/', views.sample, name="sample"),
    # path('[^]*', views.page404, name="page404"),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
