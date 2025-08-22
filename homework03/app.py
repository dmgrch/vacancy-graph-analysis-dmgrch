import ast

import dash
import networkx as nx  # type: ignore
import pandas as pd  # type: ignore
import plotly.express as px  # type: ignore
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output
from flask import Flask, render_template

from network import get_communities, plot_communities, plot_one_community

# загрузка данных
kdf = pd.read_csv("C:/Users/verit/vacancy-graph-analysis-DimaGrach1/homework03/my_title_keywords.csv")
df = kdf["title"].value_counts().reset_index()[:10]
fig = px.bar(df, x="title", y="count", title="Самые частые вакансии")

with open("graph.txt", encoding="utf-8") as f:
    edges = ast.literal_eval(f.readline())
G = nx.Graph()
G.add_edges_from(edges)
communities, community_graph = get_communities(G)

# создание стартовой страницы
server = Flask(__name__)


@server.route("/")
def index():
    return render_template("index.html")


# страница со статистикой по исходным данным
dash_dashboard_app = dash.Dash(
    __name__, server=server, url_base_pathname="/dashboard/", suppress_callback_exceptions=True
)

dash_dashboard_app.layout = html.Div(
    style={"fontFamily": "Segoe UI", "textAlign": "center", "padding": "10px", "backgroundColor": "#f0f8ff"},
    children=[
        html.H2("📊 Исходные данные"),
        html.A("← Назад", href="/", style={"color": "#28a745", "textDecoration": "none", "fontSize": "1.1em"}),
        dcc.Graph(figure=fig, style={"marginBottom": "10px", "marginTop": "10px"}),
        dash_table.DataTable(  # type: ignore
            data=kdf.to_dict("records"),
            columns=[{"name": i, "id": i} for i in kdf.columns],
            style_cell={"textAlign": "center", "padding": "1px"},
            style_header={"backgroundColor": "#28a745", "color": "white", "fontWeight": "bold"},
            style_table={"width": "100%", "margin": "0 auto"},
        ),
        html.Br(),
    ],
)


# страница с визуализацией графов
dash_dashboard_app = dash.Dash(
    __name__, server=server, url_base_pathname="/network/", suppress_callback_exceptions=True
)

dash_dashboard_app.layout = html.Div(
    [
        html.H2("📊 Визуализация вакансий по сообществам"),
        html.Div(
            [
                html.A(
                    "← Назад",
                    href="/",
                    style={
                        "color": "#28a745",
                        "textDecoration": "none",
                        "fontSize": "1.1em",
                        "fontWeight": "bold",
                    },
                )
            ],
            style={"textAlign": "left", "marginBottom": "15px"},
        ),
        dcc.Graph(
            id="all-communities-graph",
            figure=plot_communities(communities, community_graph),
            style={"marginBottom": "40px"},
        ),
        html.Label("Выберите сообщество:", style={"fontWeight": "bold"}),
        dcc.Dropdown(
            id="community-selector",
            options=[{"label": f"Сообщество {i}", "value": i} for i in range(len(communities))],  # type: ignore
            value=0,
            style={"marginBottom": "20px"},
        ),
        dcc.Graph(id="community-graph"),
        html.H4("Топ-ключевые слова в сообществе", style={"marginTop": "30px"}),
        dash_table.DataTable(  # type: ignore
            id="keyword-table",
            columns=[{"name": "Вакансия", "id": "title"}, {"name": "Ключевые слова", "id": "keywords"}],
            style_cell={"textAlign": "left", "padding": "5px"},
            style_header={"backgroundColor": "#0074D9", "color": "white", "fontWeight": "bold"},
            style_table={"width": "100%", "margin": "0 auto"},
        ),
    ],
    style={
        "fontFamily": "Segoe UI",
        "padding": "30px",
        "backgroundColor": "#f9f9f9",
        "maxWidth": "1000px",
        "margin": "0 auto",
    },
)


@dash_dashboard_app.callback(
    Output("community-graph", "figure"), Output("keyword-table", "data"), Input("community-selector", "value")
)
def update_community_graph(community_idx):
    selected_nodes = communities[community_idx]
    filtered_df = kdf[kdf["title"].isin(selected_nodes)].copy()
    table_data = filtered_df[["title", "keywords"]].to_dict("records")
    fig = plot_one_community(community_graph, selected_nodes)
    return fig, table_data


# запуск приложения
if __name__ == "__main__":
    server.run(debug=False)
