import math
import re
import string
from itertools import chain

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx  # type: ignore
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore


def preprocess_text(text: str) -> str:
    """
    Принимает на вход текст с требуемыми навыками.
    Выполняет очистку, лемматизацию и фильтрацию по части речи.
    Возвращает отфильтрованные леммы через пробел без запятых сплошным текстом.
    """
    nlp = spacy.load("ru_core_news_sm")
    punct = string.punctuation
    clean_text = [w.lower().strip(punct) for w in str(text).split()]
    clean_text = [w for w in clean_text if re.match(r"^[a-zA-Zа-яА-ЯёЁ]+(-[a-zA-Zа-яА-ЯёЁ]+)*$", w)]
    doc = nlp(" ".join(clean_text))
    lemmas = [token.lemma_ for token in doc if token.pos_ in {"NOUN", "X", "PROPN"}]
    return " ".join(lemmas)


def get_keywords(df, n_keywords=5):
    """
    Принимает на вход датафрейм с вакансиями и полем обработанных навыков.
    Возвращает датафрейм, состоящий из двух столбцов: название вакансии и столбец с ключевыми словами.
    df: входной датафрейм
    n_keywords: число ключевых слов, которое надо извлечь
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["tokens"])
    feature_names = vectorizer.get_feature_names_out()
    top_keywords = []
    for i in range(X.shape[0]):
        row = X[i].toarray().flatten()
        word_weights = [(feature_names[j], row[j]) for j in range(len(feature_names))]
        sorted_words = sorted(word_weights, key=lambda x: (-x[1], x[0]))

        keywords = [word for word, _ in sorted_words[:n_keywords]]
        top_keywords.append(keywords)

    df["keywords"] = top_keywords
    df = df[["title", "keywords"]]
    return df


def create_network(df):
    """
    Принимает на вход датафрейм с вакансиями и ключевыми словами.
    Возвращает список кортежей из пар вакансий и количества их общих ключевых слов.
    Вид кортежа внутри списка ожидается такой: (ребро1, ребро2, {'weight': вес_ребра})
    """
    edges = []

    for i in range(len(df)):
        keywords_i = set(df.loc[i, "keywords"])
        for j in range(i + 1, len(df)):
            keywords_j = set(df.loc[j, "keywords"])

            common_keywords = keywords_i.intersection(keywords_j)
            weight = len(common_keywords)

            if weight > 0:
                node1 = df.loc[i, "title"]
                node2 = df.loc[j, "title"]
                if node1 == node2:
                    continue

                edge = tuple(sorted([node1, node2])) + ({"weight": weight},)
                edges.append(edge)
    return edges


def plot_network(vac_edges):
    """
    Строит визуализацию графа с помощью matplotlib.
    """
    G = nx.Graph()
    G.add_edges_from(vac_edges)
    nx.draw(G, with_labels=False, font_weight="bold", node_size=30)
    plt.show()


def get_communities(graph):
    """
    Извлекает сообщества из графа и фильтрует их так,
    чтобы оставались только сообщества с более чем 5 узлами.
    Возвращает граф и сообщества в формате датафрейма.
    """
    communities = nx.community.louvain_communities(graph, resolution=1.2)
    comm_data = [{"n_of_nodes": len(comm), "nodes": comm} for comm in communities]
    cdf = pd.DataFrame(comm_data)
    nodes = list(chain(*cdf["nodes"].tolist()))
    S = graph.subgraph(nodes)
    communities = cdf.query("n_of_nodes>5")["nodes"].tolist()
    nodes = list(chain(*cdf.query("n_of_nodes>5")["nodes"].tolist()))
    s = S.subgraph(nodes)
    return communities, s


def create_community_node_colors(graph, communities):
    colors = list(set(mcolors.TABLEAU_COLORS.values()))
    node_colors = []
    for node in graph:
        current_community_index = 0
        for community in communities:
            if node in community:
                node_colors.append(colors[current_community_index])
                break
            current_community_index += 1
    return node_colors


def plot_communities(communities, graph):
    """
    Строит интерактивную визуализацию графа с сообществами с помощью plotly.
    """
    pos = nx.spring_layout(graph, iterations=1000, seed=30, k=3 / math.sqrt(len(graph)), scale=10.0)

    x_nodes, y_nodes = zip(*pos.values())
    edge_x, edge_y = [], []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_labels = [f"{n}<br>Degree: {graph.degree(n)}" for n in graph.nodes()]
    node_degrees = [graph.degree(n) for n in graph.nodes()]
    node_colors_list = create_community_node_colors(graph, communities)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color="gray"), mode="lines"))

    fig.add_trace(
        go.Scatter(
            x=x_nodes,
            y=y_nodes,
            mode="markers",
            marker=dict(size=[deg * 0.5 for deg in node_degrees], color=node_colors_list),
            text=node_labels,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title="Граф кластеров вакансий (все сообщества)",
        showlegend=False,
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


def plot_one_community(graph, community):
    """
    Строит график в plotly для одного сообщества.
    """
    subgraph = graph.subgraph(community)
    pos = nx.spring_layout(subgraph, iterations=300, seed=42)

    edge_x, edge_y = [], []
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x, node_y = zip(*[pos[n] for n in subgraph.nodes()])
    node_text = [f"{n}<br>Degree: {subgraph.degree(n)}" for n in subgraph.nodes()]
    degrees = [subgraph.degree(n) for n in subgraph.nodes()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="gray", width=0.5), hoverinfo="none"))

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(size=[d * 2 for d in degrees], color="skyblue"),
            text=node_text,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title="Граф сообщества",
        showlegend=False,
        plot_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return fig


if __name__ == "__main__":
    df = pd.read_csv("C:/Users/verit/vacancy-graph-analysis-DimaGrach1/homework03/python_300_vac.csv")
    df["tokens"] = df["requirement"].apply(preprocess_text)
    df = get_keywords(df)
    df.to_csv("my_title_keywords.csv", index=False)
    with open("graph.txt", "w", encoding="utf-8") as f:
        print(create_network(df), file=f)
