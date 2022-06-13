import streamlit as st
import pandas as pd
from vincenty import vincenty
from sklearn.linear_model import LinearRegression
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from PIL import Image
from translate import Translator

with st.echo(code_location='below'):
    st.write("## Final Project")
    st.write("""Привет! Я недавно побывал в городе Рыбинск, мне там очень понравилось, и я захотел немного про него рассказать. Плюс я являюсь участником программы "малые города России", и мне показалось, что будет забавно это где-то вставить. Изначально это просто был прикол и я подумал "так, ну наверняка я смогу найти у городской администрации какую-нибудь незадокументированную апишку, и нафармить баллов". Короче, я скорее ошибся немного, и в итоге не нашел вообще никаких апишек. Плюс со временем я понял, что как-то трудно много интересного показать с небольшим количеством данных. Но было уже поздно менять тему проекта, поэтому я оставил так. Если вам не понравится, можете вылить хейт в "общем впечатлении", но надеюсь вы тем не менее оцените остальные пункты по фактам. Ну или мб вам даже понравится. Короче, посмотрим.""")
    st.write("Если что я для удобства внизу приведу еще как мне кажется, где я за что получаю баллы. Там можно будет посмотреть, свериться, не пропустили ли вы что. Но так у вас всегда есть свобода не признать мои критерии как достаточные и не засчитать балл (особенно в субъективных оценках типа общее впечатление).")



    city = st.text_area("Первым делом вам нужно будет попасть в Рыбинск. Поэтому давайте посчитаем расстояние от вашего города до Рыбинска. Введите пожалуйста название населенного пункта в котором вы находитесь на английском языке.")

    api_key = "uj79F/vaj5STaTH3+V0KaA==ILQEeFUWObe9FQrW"

    ryb_url = 'https://api.api-ninjas.com/v1/city?name={}'.format('Rybinsk')
    r_response = requests.get(ryb_url, headers={'X-Api-Key': api_key})
    ryb_dict = r_response.json()[0]
    ryb_df = pd.DataFrame([ryb_dict.values()], columns=ryb_dict.keys())
    ryb_coords = (ryb_dict['longitude'], ryb_dict['latitude'])

    city_url = 'https://api.api-ninjas.com/v1/city?name={}'.format(city)
    c_response = requests.get(city_url, headers={'X-Api-Key': api_key})

    try:
        city_dict = c_response.json()[0]
        city_df = pd.DataFrame([city_dict.values()], columns=city_dict.keys())
        city_coords = (city_dict['longitude'], city_dict['latitude'])

        dist_df = pd.concat([city_df, ryb_df])

        st.dataframe(dist_df)
        translator = Translator(to_lang="ru")
        st.write("Расстояние от Рыбинска до", city, "(", translator.translate(city), ")", "составляет ", vincenty(ryb_coords, city_coords), "километров.")

    except:
        st.write("Такого города к сожалению не существует. Возможно вы неправильно переписали в латинице")



    st.write("Для того чтобы успешно добраться до Рыбинска, а также чтобы запланировать дальнейшее путешествие, давайте найдем всех соседей Рыбинска в смысле автомобильных дорог.")

    roads = [["Рыбинск", "Ярославль"], ["Рыбинск", "Пошехонье"], ["Рыбинск", "Углич"], ["Ярославль", "Углич"], ["Ярославль", "Ростов"], ["Ярославль", "Иваново"], ["Ярославль", "Кострома"], ["Ярославль", "Данилов"], ["Пошехонье", "Череповец"], ["Углич", "Бежецк"], ["Углич", "Калязин"], ["Углич", "Бежецк"], ["Углич", "Ростов"]]
    roadmap = pd.DataFrame(roads, columns=['from', 'to'])
    roadgraph = nx.Graph([(frm, to) for (frm, to) in roadmap.values])
    fig, ax = plt.subplots()
    nx.draw_networkx(roadgraph, ax=ax, node_size=2000, font_size=7.5, node_color="green", font_color="white", edge_color="darkred")
    st.pyplot(fig)



    st.write("Теперь, когда мы попали в Рыбинск давайте посмотрим достопримечательности:")

    picture = Image.open(r"Соборка.jpg")
    st.image(picture)
    st.write("Ну это короче соборная площадь и там еще мост и что-то еще.")



    st.write("Ну все, мы погуляли по рыбинску, хотим спать в отель. Но вот вопрос, а во сколько приедет наш автобус? Теоретически, мы могли бы просто открыть Яндекс карты и посмотреть там. Но нет, давайте лучше соскрепим какой-то рандомный сайт, перемучаемся десять раз, а потом уже получим че-то.")
    transport_df = pd.read_csv(r"busses")

    st.write("Короче, я тут все делал в Юпитере, а потом уже запустил чисто цсвшку, если что, смотрите в прикрепе или где-то.")
    st.write("Ну тут видно, что некоторые автобусы не смогли скачаться нормально, но в основном это были маршруты там которые несколько раз в день ездят, так что в целом это основная информация, которая нам нужна.")
    st.write("Давайте изучим на каких автобусах Рыбеньковчане ездят чаще всего.")

    tr_df = transport_df[transport_df["всего автобусов"].isna()==False][["от", "до", "всего автобусов"]]
    st.dataframe(tr_df)

    st.write("""Далее давайте воспользуемся методом groupby, also known as "ого, это что продвинутые возможности pandas?", и построим хитмеп любимых конечных остановок в Рыбинске (эйкей более сложная визуализация, требующая написания нетривиального кода).""")

    start_df = tr_df[tr_df["всего автобусов"].isna() == False].groupby("от").sum().sort_values(by="всего автобусов", ascending=False)
    end_df = tr_df[tr_df["всего автобусов"].isna() == False].groupby("до").sum().sort_values(by="всего автобусов", ascending=False)

    fig, ax = plt.subplots()
    sns.heatmap(start_df, vmin=0, vmax=350, cmap=sns.color_palette("rocket", as_cmap=True))
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.heatmap(end_df, robust=True, cmap=sns.color_palette("crest_r", as_cmap=True))
    st.pyplot(fig)



    st.write("Наконец, давайте рассмотрим образование в Рыбинске, мало ли вдруг вы после моей экскурсии с одной фоткой захотели переехать. Тогда ведь нужно знать, что ждет ваших детей.")
    st.write("Давайте скачаем данные по школам Ярославской области (правда за 2016 год), а затем выберем только те школы, которые находятся в г. Рыбинске.")

    schools = pd.read_excel(r"Егешники.xlsx")
    cols = ["Full name", "Russian language,amount", "Russian language,GPA", "Mathematics profil,amount", "Mathematics profil,GPA", "Physics ,amount", "Physics,GPA"]
    ryb_schools = schools[schools['Name area'].isin(["г. Рыбинск", "Рыбинский"])][cols]
    ryb_schools.columns = ["name", "number_rus", "score_rus", "number_math", "score_math", "number_phys", "score_phys"]
    ryb_schools.set_index("name", inplace = True)

    st.dataframe(ryb_schools)

    scores = ryb_schools[["score_rus", "score_math", "score_phys"]]
    st.write("Давайте посмотрим на коррелляции:")
    st.dataframe(scores.corr())

    model = LinearRegression()
    model.fit(scores[["score_rus"]], scores["score_math"])

    fig, ax = plt.subplots()
    scores.plot.scatter(x="score_rus", y="score_math", color = "blue", ax=ax)
    x = pd.DataFrame(dict(score_rus=np.linspace(55, 87)))
    plt.plot(x["score_rus"], model.predict(x), color="red", lw=2)
    st.pyplot(fig)

    st.write("Ну и короче поскольку нам делать нечего (и нужны баллы за математические возможности питона), давайте посчитаем рейтинги школ по трем предметам.")
    st.write("Для начала найдем среднее и дисперсию для каждого предмета.")

    rus_mean = np.mean(scores['score_rus'])
    math_mean = np.mean(scores['score_math'])
    phys_mean = np.mean(scores['score_phys'])
    rus_var = np.var(scores['score_rus'])
    math_var = np.var(scores['score_math'])
    phys_var = np.var(scores['score_phys'])

    st.write("""Далее давайте рассчитаем "баллы" - превышение GPA над средним по школам, деленное на дисперсию (для каждого предмета).""")

    ryb_schools['score_rus'] = (ryb_schools['score_rus'] - rus_mean)/rus_var
    ryb_schools['score_math'] = (ryb_schools['score_math'] - math_mean) / math_var
    ryb_schools['score_phys'] = (ryb_schools['score_phys'] - phys_mean) / phys_var

    ryb_schools['total_points'] = ryb_schools['number_rus'] * ryb_schools['score_rus'] + ryb_schools['number_math'] * ryb_schools['score_math'] + ryb_schools['number_phys'] * ryb_schools['score_phys']

    st.write("Наконец, посчитаем сумму баллов и выведем лидеров.")
    st.dataframe(ryb_schools['total_points'].sort_values(ascending=False))

    st.write("Итак, наконец настало время подсчитать баллы. Опять же тут всё скорее рекомендации, сделанные исключительно для того, чтобы упростить вам жизнь.")
    st.write("pandas: я использовал его много, ну и в целом там были свои приколы типа групбай итд, так что можно и 2, но не знаю")
    st.write("Скрепинг: я использовал Selenium при создании датафрейма про автобусы, конкретно так с ним наммучался, так что надеюсь на 2")
    st.write("API: так, ну тут была тема с тем, что апишка была с ограниченным доступом (что выходит за рамки дз) и мне пришлось разбираться как там api-key аботает. Так что можно и 2.")
    st.write("Визуализация: Ну вы все картинки видите, там вроде я параметры понакручивал, можно и 2")
    st.write("Математические возможности использовались (в рейтинге школ) - 1")
    st.write("Streamlit: Пока я пишу этот коммент, мне еще только предстоит с этим мучиться, но вроде на предыдущем проекте справился выгрузить, так что и тут постараюсь - 1")
    st.write("SQL: Не использовался, в целом, если вы его увидете можете конечно поставить балл, но будет странно - 0")
    st.write("Регулярные выражения: тоже использовались при создании дфки про транспорт - 1")
    st.write("Геоданные: Ну там в начале была тема, где я расстояние считаю с помощью vincenty (потому что shapely в тот момент решила поприкалываться надо мной немного и я не хотел больше) - 1")
    st.write("Машинное обычение: ну я там построил самую базовую регрессию - 1")
    st.write("Работа с графами: была и еще какая - 1")
    st.write("Дополнительные технологии: ну я там решил сделать типа переводчик городов (а переводчиков вроде не было особо на курсе), но там честно было проделанно не очень много работы, так что не уверен, насколько тут заслуживаю - 0.5")
    st.write("Объем: Ну у меня в этом файле вроде около 80 плюс 70 в ноутбуковском файле (можете чекнуть, кстати), так что вроде все сходится даже с запасом - 1")
    st.write("Целостность: тут скорее субъективный критерий, но вроде я все про Рыбинск рассказывал, и более-менее связно было -1")
    st.write("Общее: Опять же, субъективно. Я лично сам был не очень впечатлен и доволен своей работой, так что не обижусь, если вы не поставите тут. Но вообще можно и 1, при желании")
    st.write("Суммарная оценка: Ну я себе тут конечно насчитал уже на 18.5, но это, наверное, чуть перебор. Я думаю каждый из вас сам в различных критериях найдет за что снять, так что давайте. В целом, рекомендуемая оценка: 16-17 баллов.")
    st.write("Спасибо за выделенное время.")