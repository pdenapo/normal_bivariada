
# Programa para mostrar la normal bivariada interactivamente usando plotly y dash

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
    def maxima_densidad(self):
        return self.distribucion.pdf([0,0])

#Tamaño del cuadrado en el que vamos a trabajar
R=5

# Definimos la grilla que vamos a usar
x = np.linspace(-R, R, num=100)
y = np.linspace(-R, R, num=100)
X, Y = np.meshgrid(x,y)

# 	https://www.htmlsymbols.xyz/unicode/
sigma_en_html='\u03C3'                   
rho_en_html='\u03C1'

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
# Usamos uno de los temas de default de Bootstrap https://bootswatch.com/cyborg/
#app = dash.Dash(external_stylesheets=[dbc.themes.CYBORG])

server = app.server

import plotly.graph_objects as go

app.layout =  dbc.Container(
html.Div(children=[
    html.H1(children='La distribución normal bivariada', className="mt-5 text-center"),

    html.H2(children='Un ejemplo importante de distribución de un vector aleatorio',className="pt-3 text-center"), 
    html.H2(children='con aplicaciones en estadística.', className="pt-1 text-center"),

        html.Div([
        dcc.Slider(
        id='sigma_x-slider',
        min=0,
        max=3,
        step=0.1,
        value=1
        ),     
        html.Div(id='sigma_x-slider-output-container',className="text-center")]),

         html.Div([
        dcc.Slider(
        id='sigma_y-slider',
        min=0,
        max=3,
        step=0.1,
        value=1
        ),     
        html.Div(id='sigma_y-slider-output-container',className="text-center")]),

         html.Div([
        dcc.Slider(
        id='rho-slider',
        min=0,
        max=1,
        step=0.05,
        value=0
        ),     
        html.Div(id='rho-slider-output-container',className="text-center")]),



    # Para que el código funcione es fundamental darle un valor inicial a value!
    dcc.Checklist(
    options=[
        {'label': 'Contornos en x', 'value': 'X'},
        {'label': 'Contornos en y', 'value': 'Y'},
        {'label': 'Contornos en z', 'value': 'Z'}
    ], id='my-checkbox', value=[]),

    #html.Div(id='checkbox-output'),

    dcc.Graph(id='my-graph')
]))

#@app.callback(
#    Output(component_id='checkbox-output', component_property='children'),
#    Input(component_id='my-checkbox', component_property='value')
#)
#def update_output_div(input_value):
#    return 'Output: {}'.format(input_value)

@app.callback(
    Output('sigma_x-slider-output-container', 'children'),
    Input('sigma_x-slider', 'value')
)
def update_output(value):
    return sigma_en_html+'_x='+format(value)

@app.callback(
    Output('sigma_y-slider-output-container', 'children'),
    Input('sigma_y-slider', 'value')
)
def update_output(value):
    return sigma_en_html+'_y='+format(value)


@app.callback(
    Output('rho-slider-output-container', 'children'),
    Input('rho-slider', 'value')
)
def update_output(value):
    return rho_en_html+'='+format(value)


def update_output(value):
    return sigma_en_html+'_y='+format(value)    

@app.callback(
    Output(component_id='my-graph', component_property='figure'),
    Input(component_id='my-checkbox', component_property='value'),
    Input('sigma_x-slider', 'value'),
    Input('sigma_y-slider', 'value'),
    Input('rho-slider', 'value')
)
def update_figure(check_box_value,sigma_x,sigma_y,rho):
    mi_distribucion=normal_bivariada(float(sigma_x),float(sigma_y),float(rho))
    mi_distribucion.evaluar_densidad_en_una_grilla(X,Y)
    fig = go.Figure(go.Surface(
    contours = {
         "z": {"show": 'Z' in check_box_value , "start": 0.0, "end": mi_distribucion.maxima_densidad(), "size": 0.01,"color":"white"},
         "x": {"show": 'X' in check_box_value, "start": -R, "end": R, "size": 0.5,},
         "y": {"show": 'Y' in check_box_value, "start": -R, "end": R, "size": 0.5}
    },
    x = x,
    y = y,
    z = mi_distribucion.densidad))
    fig.update_layout(
        scene = {
            "xaxis": {"nticks": 20},
            "yaxis": {"nticks": 20},
            "zaxis": {"nticks": 4,'range':[-0.1, mi_distribucion.maxima_densidad()*1.2] },
            'camera_eye': {"x": 0, "y": -1, "z": 0.5},
            "aspectratio": {"x": 1, "y": 1, "z": 1}
        })
    return fig

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Caminata al azar')
    parser.add_argument("--debug", action='store_true', help='debug')
    args = parser.parse_args()
    app.run_server(debug=args.debug)