# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

#import plotly.express as px
#import pandas as pd

import numpy as np
from scipy.stats import multivariate_normal

class normal_bivariada:
    def __init__(self,sigma_x,sigma_y,rho):
    # Consideramos la esperanza como vector 
        self.sigma_x= sigma_x
        self.sigma_y= sigma_y
        self.rho= rho
        mu = np.array([0.0,0.0])
        var_x = sigma_x**2
        var_y = sigma_y**2
        cov_xy= sigma_x*sigma_y*rho
        Sigma = np.array([[var_x, cov_xy ], [cov_xy, var_y]])
        self.distribucion= multivariate_normal(cov = Sigma, mean = mu)
    def evaluar_densidad_en_una_grilla(self,X,Y):
        # Calculamos la función de densidad en cada funto de la grilla
        self.densidad = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                self.densidad[i,j] = self.distribucion.pdf([X[i,j], Y[i,j]])
    def generar_al_azar(self,tamaño):
        return self.distribucion.rvs(size = tamaño)
    
# Definimos la grilla que vamos a usar
x = np.linspace(-3, 3, num=100)
y = np.linspace(-3, 3, num=100)
X, Y = np.meshgrid(x,y)
                   
mi_distribucion=normal_bivariada(1,1,0)
mi_distribucion.evaluar_densidad_en_una_grilla(X,Y)
mi_distribucion.distribucion.pdf([0.0,0.0])

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
# Usamos uno de los temas de default de Bootstrap https://bootswatch.com/cyborg/
#app = dash.Dash(external_stylesheets=[dbc.themes.CYBORG])

import plotly.graph_objects as go

app.layout =  dbc.Container(
html.Div(children=[
    html.H1(children='La distribución normal bivariada', className="mt-5 text-center"),

    html.H2(children='Un ejemplo importante de distribución de un vector aleatorio',className="pt-3 text-center"), 
    html.H2(children='con aplicaciones en estadística.', className="pt-1 text-center"),

    # Para que el código funcione es fundamental darle un valor inicial a value!
    dcc.Checklist(
    options=[
        {'label': 'Contornos en x', 'value': 'X'},
        {'label': 'Contornos en y', 'value': 'Y'},
        {'label': 'Contornos en z', 'value': 'Z'}
    ], id='my-checkbox', value=[]),

    html.Div(id='my-output'),

    dcc.Graph(id='my-graph')
]))

@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='my-checkbox', component_property='value')
)
def update_output_div(input_value):
    return 'Output: {}'.format(input_value)

@app.callback(
    Output(component_id='my-graph', component_property='figure'),
    Input(component_id='my-checkbox', component_property='value')
)
def update_figure(input_value):
    print(input_value)
    fig = go.Figure(go.Surface(
    contours = {
         "z": {"show": 'Z' in input_value , "start": 0.0, "end": 0.20, "size": 0.01,"color":"white"},
         "x": {"show": 'X' in input_value, "start": -3.0, "end": 3.0, "size": 0.5,},
         "y": {"show": 'Y' in input_value, "start": -3.0, "end": 3.0, "size": 0.5}
    },
    x = x,
    y = y,
    z = mi_distribucion.densidad))
    fig.update_layout(
        scene = {
            "xaxis": {"nticks": 20},
            "yaxis": {"nticks": 20},
            "zaxis": {"nticks": 4,'range':[-0.1, 0.2] },
            'camera_eye': {"x": 0, "y": -1, "z": 0.5},
            "aspectratio": {"x": 1, "y": 1, "z": 1}
        })
    return fig

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Caminata al azar')
    parser.add_argument("--debug", action='store_true', help='denug')
    args = parser.parse_args()
    app.run_server(debug=args.debug)