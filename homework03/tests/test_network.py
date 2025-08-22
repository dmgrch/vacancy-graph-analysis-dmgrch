import unittest
import pandas as pd  # type: ignore
import networkx as nx  # type: ignore

from network import create_network, get_keywords


DF = pd.read_csv("title_tokens.csv")
KDF = pd.read_csv("title_keywords.csv")


class NetworkTestCase(unittest.TestCase):
    def test_create_network(self):
        df = pd.DataFrame([["data analyst",
                            "data scientist",
                            "web developer",
                            "backend developer"]], index=["title"]).T

        df["keywords"] = [["python", "bi systems", "dataviz", "flask"],
                          ["python", "pytorch", "machine learning", "dataviz", "bi systems"],
                          ["django", "vue.js", "flask"],
                          ["python", "django", "flask"]]

        expected_edges = [
            ("backend developer", "data analyst", {"weight": 2}),
            ("backend developer", "data scientist", {"weight": 1}),
            ("backend developer", "web developer", {"weight": 2}),
            ("data analyst", "data scientist", {"weight": 3}),
            ("data analyst", "web developer", {"weight": 1}),
        ]
        edges = create_network(df)
        self.assertCountEqual(expected_edges, edges)


    def test_get_keywords(self):
        df_with_keywords = get_keywords(DF, 5)
        df_with_keywords["str_keywords"] = df_with_keywords.iloc[:,1].apply(lambda x: ", ".join(x))

        self.assertTrue(df_with_keywords.iloc[:,[0,2]].equals(KDF))


    def test_pipeline(self):
        df_with_keywords = get_keywords(DF, 5)
        edges = create_network(df_with_keywords)

        net = nx.Graph()
        net.add_edges_from(edges)

        self.assertEqual(len(net.nodes), 205)
        self.assertEqual(len(net.edges), 1505)
