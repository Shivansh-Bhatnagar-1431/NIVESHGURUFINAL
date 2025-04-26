from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('blogs/', views.blog_list, name='blog_list'),
    path('my-blogs/', views.my_blogs, name='my_blogs'),
    path('create-blog/', views.create_blog, name='create_blog'),
    
    
]