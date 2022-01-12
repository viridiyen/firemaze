import PySimpleGUI as sg

dim = 5

#create board of the right dimensions
#color in obstacles black
layout = [
    [sg.Graph(canvas_size=(400, 400), graph_bottom_left=(0,0), graph_top_right=(400, 400), background_color='red', key='graph')],
    [sg.T('Change circle color to:'), sg.Button('Red'), sg.Button('Blue'), sg.Button('Move')]
    ]      

window = sg.Window('Graph test', layout)      
window.Finalize()      

graph = window['graph']   
#circle = graph.DrawCircle((75,75), 25, fill_color='black',line_color='white')      
#point = graph.DrawPoint((75,75), 10, color='green')      
#oval = graph.DrawOval((25,300), (100,280), fill_color='purple', line_color='purple')      
rectangle = graph.DrawRectangle((50,300), (100,280), line_color='purple')      
#line = graph.DrawLine((0,0), (100,100))      

num_rectangles = 400/dim

while True:
    event, values = window.read(timeout=10)      
    if event == sg.WIN_CLOSED:      
        break  
    
    '''
    if event is 'Blue':      
        graph.TKCanvas.itemconfig(circle, fill = "Blue")      
    elif event is 'Red':      
        graph.TKCanvas.itemconfig(circle, fill = "Red")  
    '''
    if event is 'Blue':   
       # graph.MoveFigure(point, 10,10)      
       # graph.MoveFigure(circle, 10,10)      
        #graph.MoveFigure(oval, 10,10)      
        graph.TKCanvas.itemconfig(rectangle, fill = "Blue")  
    elif event is 'Red':      
        graph.TKCanvas.itemconfig(rectangle, fill = "Red")
    
    