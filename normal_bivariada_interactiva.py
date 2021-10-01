
# Programa para mostrar la normal bivariada interactivamente usando plotly y dash.
# (C) 20021 Pablo De Nápoli
# Se distribuye bajo la licencia MIT.

import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

import plotly.express as px
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
    def evaluar_densidad_en_una_grilla(self,grilla):
        # Calculamos la función de densidad en cada funto de la grilla
        self.densidad=self.distribucion.pdf(np.dstack(grilla))
    def maxima_densidad(self):
        return self.distribucion.pdf([0,0])
    def generar_al_azar(self,tamaño,semilla):
        np.random.seed(semilla)
        return self.distribucion.rvs(size = tamaño)

#Tamaño del cuadrado en el que vamos a trabajar
R=7
# numero de puntos por lado de la grilla
N=60

import time
mi_semilla= np.random.seed(int(np.ceil(time.time())))

# Definimos la grilla que vamos a usar
x = np.linspace(-R, R, num=60)
y = np.linspace(-R, R, num=60)
grilla = np.meshgrid(x,y)

# 	https://www.htmlsymbols.xyz/unicode/
sigma_en_html='\u03C3'                   
rho_en_html='\u03C1'

app = dash.Dash(external_stylesheets=[dbc.themes.FLATLY],title='Normal Bivariada')
# Usamos uno de los temas de default de Bootstrap https://bootswatch.com/flatly
# https://dash-bootstrap-components.opensource.faculty.ai/docs/themes/

server = app.server

import plotly.graph_objects as go

def title():
    return html.H1(children='La distribución normal bivariada', className="mt-5 text-center")

def sigma_x_slider():
    return html.Div([
        html.Div(id='sigma_x-slider-output-container',className="text-center"),
        dcc.Slider(
        id='sigma_x-slider',
        min=0,
        max=3,
        step=0.1,
        value=1)
        ])

@app.callback(
    Output('sigma_x-slider-output-container', 'children'),
    Input('sigma_x-slider', 'value')
)
def update_output_sigma_x(value):
    return "Desviación estándar de x "+ sigma_en_html+'_x='+format(value)

def sigma_y_slider():
    return html.Div([
        html.Div(id='sigma_y-slider-output-container',className="text-center"),
        dcc.Slider(
        id='sigma_y-slider',
        min=0,
        max=3,
        step=0.1,
        value=1)
        ])

@app.callback(
    Output('sigma_y-slider-output-container', 'children'),
    Input('sigma_y-slider', 'value')
)
def update_output_sigma_y(value):
    return "Desviación estándar de y "+ sigma_en_html+'_y='+format(value)

def rho_slider():
    return html.Div([
        html.Div(id='rho-slider-output-container',className="text-center"),
        dcc.Slider(
        id='rho-slider',
        min=0,
        max=1,
        step=0.05,
        value=0)
        ])
@app.callback(
    Output('rho-slider-output-container', 'children'),
    Input('rho-slider', 'value')
)
def update_output_rho(value):
    return "Coeficiente de correlación "+ rho_en_html+'='+format(value)        

def tipo_de_grafico_radio_items():
    return dcc.RadioItems(
        options=[
            {'label': 'Superficie', 'value': 'S'},
            {'label': 'Contornos', 'value': 'C'},
            {'label': 'Puntos al azar', 'value': 'A'}
        ],
        style = {'text-align': 'center'}, 
        className = 'dcc_compon pt-3 text-center',
        value='S',
        id='tipo-de-grafico-radio-items'
    )

def contornos_check_list():
    return dcc.Checklist(
    options=[
        {'label': 'Contornos en x', 'value': 'X'},
        {'label': 'Contornos en y', 'value': 'Y'},
        {'label': 'Contornos en z', 'value': 'Z'}
    ], 
    id='my-checkbox', 
    value=[],
    style = {'display': 'block'}, 
    className = 'dcc_compon pt-3 text-center',
    )

@app.callback(
    Output(component_id='my-checkbox', component_property='style'),
    Input(component_id='tipo-de-grafico-radio-items', component_property='value')
)
def show_hide_element(value):
    if value == 'S':
        return {'display': 'block'}
    else:
        return {'display': 'none'}

def cuantos_puntos_slider():
    return html.Div([
        html.Div(id='cuantos-puntos-output-container',className="text-center"), 
        dcc.Slider(
        id='cuantos-puntos-slider',
        min=1,
        max=5000,
        step=1,
        value=100),
        ],id='cuantos-puntos-slider-div')

#,style={'display': 'block'},

@app.callback(
    Output('cuantos-puntos-output-container', 'children'),
    Input('cuantos-puntos-slider', 'value')
)
def update_output_cuantos_puntos(value):
    return "cuantos puntos="+format(value)

@app.callback(
    Output(component_id='cuantos-puntos-slider-div', component_property='style'),
    Input(component_id='tipo-de-grafico-radio-items', component_property='value')
)
def show_hide_cuantos_puntos(value):
    if value == 'A':
        return {'display': 'block'}
    else:
        return {'display': 'none'}

def gráfico():
   return html.Div(
        dcc.Graph(id='my-graph'),
        className = 'dcc_compon center-block')

@app.callback(
    Output(component_id='cuantos-slider', component_property='style'),
    Input(component_id='tipo-de-grafico-radio-items', component_property='value')
)
def show_hide_cuantos(value):
    if value == 'A':
        return {'display': 'block'}
    else:
        return {'display': 'none'}

@app.callback(
    Output(component_id='my-graph', component_property='figure'),
    Input(component_id='my-checkbox', component_property='value'),
    Input(component_id='sigma_x-slider', component_property='value'),
    Input(component_id='sigma_y-slider', component_property='value'),
    Input(component_id='rho-slider', component_property='value'),
    Input(component_id='tipo-de-grafico-radio-items', component_property='value'),
    Input(component_id='cuantos-puntos-slider', component_property='value')
)
def update_figure(check_box_value,sigma_x=1,sigma_y=1,rho=0,tipo_de_grafico='S',cuantos_puntos=100):
    print("update_figure llamado")
    mi_distribucion=normal_bivariada(float(sigma_x),float(sigma_y),float(rho))
    if (tipo_de_grafico=='S'):
        mi_distribucion.evaluar_densidad_en_una_grilla(grilla)
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
    elif (tipo_de_grafico=='C'):
        mi_distribucion.evaluar_densidad_en_una_grilla(grilla)
        fig = go.Figure(go.Contour(
        x = x,
        y = y,
        z = mi_distribucion.densidad))
    else:
        datos=  mi_distribucion.generar_al_azar(int(cuantos_puntos),mi_semilla)
        fig = px.scatter(x=datos[:,0],y=datos[:,1],trendline="ols", trendline_color_override="red",range_x=[-6,6],range_y=[-6,6], width=600, height=600)
    return fig

app.layout =  dbc.Container(
html.Div(children=[
        title(),
        #html.H2('hola'),     
        sigma_x_slider(),
        sigma_y_slider(),
        rho_slider(),
        tipo_de_grafico_radio_items(),
        contornos_check_list(),
        cuantos_puntos_slider(),
        gráfico()
]))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Bistribución Normal Bivariada')
    parser.add_argument("--debug", action='store_true', help='debug')
    args = parser.parse_args()
    app.run_server(debug=args.debug)