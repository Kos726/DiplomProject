"""
URL configuration for Diplom_726 project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
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
from appbase.views import platform, objective
from appbase.views import sign_up_by_django #, sign_up_by_html
from appbase.views import lib_dataset, lib_methods, news_page, results


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', platform),
    path('objective/', objective),
    # path('sign_up_by_html/', sign_up_by_html),
    path('django_sign_up/', sign_up_by_django),
    path('platform/news/', news_page),
    path('lib_dataset/', lib_dataset),
    path('lib_methods/', lib_methods),
    path('results/', results),

]
