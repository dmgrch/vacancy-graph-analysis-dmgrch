# https://api.hh.ru/openapi/redoc#section/Obshaya-informaciya

import pandas as pd  # type: ignore

import session  # type: ignore


def get_vacancies(pages, tag):
    """
    Функция принимает на вход количество страниц, с которых надо собрать информацию (pages)
    и ключевое слово, по которому должен осуществляться поиск (tag).
    Вернуть необходимо список уникальных по айди вакнсий.
    """
    url = "/vacancies"
    base_url = "https://api.hh.ru"
    s = session.Session(base_url, timeout=3)

    vacancies = []
    for page in range(pages):
        headers = {"User-Agent": "example@yandex.ru"}
        params = {"page": page, "text": tag}

        response = s.get(url, headers=headers, params=params)
        data = response.json()
        for vacancy in data["items"]:
            item = (
                vacancy["id"],
                vacancy["name"],
                vacancy["snippet"]["requirement"],
                vacancy["snippet"]["responsibility"],
            )
            vacancies.append((item))
    return vacancies


if __name__ == "__main__":
    vacancies = get_vacancies(5, "python")
    df = pd.DataFrame(vacancies, columns=["id", "title", "requirement", "responsibility"])
    df.to_csv("python_300_vac.csv", index=False)
