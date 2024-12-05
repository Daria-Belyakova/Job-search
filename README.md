Test&control_groups_comparisons: 
Данные - количество кликов в разных исследуемых группах.
1 код представляет обработку данных на гомоскедастичность. Данные проверяются на нормальность. 
Затем данные разбиваются и также проверяются на нормальность
Следующим шагом является проверка влияния фактора групп на количество кликов. Используются различные статитстические методы для сравнения количества кликов различных групп.

2 код предоставляет уже большое количество факторов, которые могут влиять на изменения в данных. 
На графике мы смотрим распределение событий для тестовой и контрольной групп. 
Исходя из количества факторов, мы смогли выявить три независимые переменные, которые могут влиять на зависимую (segment, group и совместное влияние факторов segment и group).


Taxi_analysis:
Анализ различных баз данных


Price predicts (cars):
Задача предсказать изменение цены при изменении показателя horsepower, используя мождель с одним фактором и со множеством.
Изначально необходимо было оставить только марки автомобилей, потому что в базе данных хранились модели. 
Так как названия марок вносились вручную, неправильные названия были переименованы.
Была построена таблица корреляции признаков, чтобы посмотреть, как они зависимы между собой.
Так как некоторые данные являются катеогриальными, был применен метод get_dummies.
После этого создаем модель влияния переменной horsepower на изменение цены.
Также создаем модель влияния всех факторов и модель влияния всех факторов без фактора марки автомобиля на изменение цены.
Для последней добавляем константу и строим модель для получения значимостей предикторов.
Для наглядности выводим сводную таблицу с результатами регрессии, процент объясненной дисперсии и количество незначимых предикторов.
Мы видим, что у последней модели лучше объяснена дисперсия, поэтому мы используем ее результаты для предсказания изменения цены при единчном изменении фактора horsepower
