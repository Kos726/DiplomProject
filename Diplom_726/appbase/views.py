from django.shortcuts import render
from django.http import HttpResponse
from .forms import UserRegister
from .models import *
from django.core.paginator import Paginator


# Create your views here.
def platform(request):
    title = "Главная страница"

    context = {
        'title': title,
    }

    return render(request, 'base/platform.html', context)


def objective(request):
    title = "Общая информация"

    context = {
        'title': title,
    }

    return render(request, 'base/objective.html', context)


info_ = {}

"""
# Create your views here.
def sign_up_by_html(request):
    title = 'Регистрация'
    button_back = "На главную"
    if request.method == "POST":
        # Получение данных
        username = request.POST.get("username")
        password = request.POST.get("password")
        repeat_password = request.POST.get("repeat_password")
        age = request.POST.get("age")
        # HTTP ответ пользователя

        # Валидация данных
        buyer_find = Buyer.objects.filter(slug=username).count()
        print(buyer_find)
        is_valid_age = int(age) >= 18
        is_valid_user = buyer_find == 0  # если нашел, то >0, значит False
        is_valid_password = password == repeat_password
        print(is_valid_user)
        if is_valid_age is False:
            info_.update({'error': 'Вы должны быть старше 18'})

        elif is_valid_user is False:
            info_.update({'error': 'Пользователь уже существует'})

        elif is_valid_password is False:
            info_.update({'error': 'Пароли не совпадают'})

        # если все проверки пройдены
        elif is_valid_age and is_valid_user and is_valid_password is True:
            Buyer.objects.create(name=username, balance=0, age=age)
            return HttpResponse(f'Приветствуем, {username}!')  # render(request, 'base/platform.html', context)

        # Если GET

    context_ = {
        'title': title,
        'button_back': button_back,
    }

    context = {**context_, **info_}
    print(context)
    return render(request, 'base/registration_page.html', context)

"""
def sign_up_by_django(request):
    if request.method == "POST":
        # Получение данных
        form = UserRegister(request.POST)
        if form.is_valid():
            # Обработка данных формы
            username = form.cleaned_data["username"]
            password = form.cleaned_data["password"]
            repeat_password = form.cleaned_data["repeat_password"]
            age = form.cleaned_data["age"]

        # Валидация данных
        buyer_find = Buyer.objects.filter(slug=username.lower()).count()
        is_valid_age = int(age) >= 18
        is_valid_user = buyer_find == 0  # если нашел, то >0, значит False
        is_valid_password = password == repeat_password

        if is_valid_age is False:
            info_.update({'error': 'Вы должны быть старше 18'})

        elif is_valid_user is False:
            info_.update({'error': 'Пользователь уже существует'})

        elif is_valid_password is False:
            info_.update({'error': 'Пароли не совпадают'})

        # если все проверки пройдены
        elif is_valid_age and is_valid_user and is_valid_password is True:

            Buyer.objects.create(name=username, balance=0, age=age)

            info_.update({'error': f'Приветствуем, {username}!'})
            # return HttpResponse(f'Приветствуем, {username}!')

        # HTTP ответ пользователя
    else:
        form = UserRegister()

        # Если GET
    form_ = {'form': form}
    context = {**form_, **info_}
    return render(request, 'base/registration_page.html', context)


def news_page(request):
    title = "Новости"
    button_back = "Вернуться обратно"

    # получаем все посты
    news = News.objects.all()

    # создаем пагинатор
    paginator = Paginator(news, 1)  # 10 постов на странице

    # получаем номер страницы, на которую переходит пользователь
    page_number = request.GET.get('page')

    # получаем посты для текущей страницы
    page_news = paginator.get_page(page_number)

    # передаем контекст в шаблон

    page_form = {
        'title': title,
        'button_back': button_back,
    }

    page_news_ = {'page_news': page_news}
    context = {**page_form, **page_news_}
    return render(request, 'base/news.html', context)


def lib_dataset(request):
    title = "Библиотека датасетов"
    button_back = "Вернуться обратно"

    # получаем все посты
    datasets = Dataset_Lib.objects.all()

    # создаем пагинатор
    paginator = Paginator(datasets, 5)  # 10 постов на странице

    # получаем номер страницы, на которую переходит пользователь
    page_number = request.GET.get('page')

    # получаем посты для текущей страницы
    page_datasets = paginator.get_page(page_number)

    # передаем контекст в шаблон

    page_form = {
        'title': title,
        'button_back': button_back,
    }

    page_datasets_ = {'page_datasets': page_datasets}
    context = {**page_form, **page_datasets_}
    return render(request, 'base/lib_dataset.html', context)


def lib_methods(request):
    title = "Библиотека методов ML"
    button_back = "Вернуться обратно"

    # получаем все посты
    methods = MethodML.objects.all()

    """
    # создаем пагинатор
    paginator = Paginator(methods, 5)  # 10 постов на странице

    # получаем номер страницы, на которую переходит пользователь
    page_number = request.GET.get('page')

    # получаем посты для текущей страницы
    page_methods = paginator.get_page(page_number)
    """
    # передаем контекст в шаблон
    page_form = {
        'title': title,
        'button_back': button_back,
    }

    methods_ = {'methods': methods}
    context = {**page_form, **methods_}
    return render(request, 'base/lib_methods.html', context)


def results(request):
    title = "Тестированиеб модели и рекомендации"

    context = {
        'title': title,
    }

    return render(request, 'base/results.html', context)
