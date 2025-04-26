from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib import messages
from django.contrib.auth.forms import UserCreationForm
import logging
from django.contrib.auth.decorators import login_required
from django.db.models import Q
from .models import TradeBlog
from .forms import BlogForm

@login_required
def blog_list(request):
    search_query = request.GET.get('search', '')
    blogs = TradeBlog.objects.filter(
        Q(title__icontains=search_query) | 
        Q(content__icontains=search_query)
    ).order_by('-created_at')  # Order by created_at, descending (latest first)
    return render(request, 'main/blog_lists.html', {'blogs': blogs})

@login_required
def my_blogs(request):
    blogs = TradeBlog.objects.filter(author=request.user)
    return render(request, 'main/my_blogs.html', {'blogs': blogs})

@login_required
def create_blog(request):
    if request.method == 'POST':
        form = BlogForm(request.POST, request.FILES)
        if form.is_valid():
            blog = form.save(commit=False)
            blog.author = request.user
            blog.save()
            return redirect('my_blogs')
    else:
        form = BlogForm()
    return render(request, 'main/create_blog.html', {'form': form})
# Get an instance of a logger
logger = logging.getLogger(__name__)

def home(request):
    return render(request, 'main/home.html')





# main/views.py
def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, f'Welcome back, {username}!')
            return redirect('home')
        else:
            messages.error(request, 'Invalid username or password')
    return render(request, 'main/login.html')

def logout_view(request):
    username = request.user.username
    logout(request)
    messages.info(request, 'You have been logged out successfully')
    return redirect('home')

def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Account created successfully!')
            return redirect('home')
    else:
        form = UserCreationForm()
    return render(request, 'main/signup.html', {'form': form})

# dashboard view work
@login_required
def dashboard_view(request):
   
    return render(request, 'nse_dashboard/dashboard.html')
@login_required
def lstm_model(request):
   
    return render(request, 'main/lstm.html')

