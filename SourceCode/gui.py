# -*- encoding: utf-8 -*-
"""
@File    : gui.py
@Time    : 11/15/2021 9:01 AM
@Author  : Liuwen Zou
@Email   : levinforward@163.com
@Software: PyCharm
"""
import base64
import io
import os
import time
import PIL
import PySimpleGUI as sg
import cv2
from dask.tests.test_base import np
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import imutils
import tools

matplotlib.use('TkAgg')


img_key = None
BPAD_LEFT_INSIDE = (0, 10)
DARK_HEADER_COLOR = '#1B2838'
toolbar_width = 1900
toolbar_height = 50
original_img_arr = np.zeros((1, 1))
original_img_arr_normalized = np.zeros((1, 1))
original_img_name = ''
temp_original_img_arr = np.zeros((1, 1))
temp_original_img_arr_normalized = np.zeros((1, 1))
cur_img_arr = np.zeros((1, 1))
cur_img_arr_normalized = np.zeros((1, 1))
last_event = None
pre_degree = 0


def convert_to_bytes(file_or_bytes, resize=None):

    if isinstance(file_or_bytes, str):
        img = PIL.Image.open(file_or_bytes)
    else:
        try:
            img = PIL.Image.open(io.BytesIO(base64.b64decode(file_or_bytes)))
        except Exception as e:
            dataBytesIO = io.BytesIO(file_or_bytes)
            img = PIL.Image.open(dataBytesIO)

    cur_width, cur_height = img.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height/cur_height, new_width/cur_width)
        img = img.resize((int(cur_width*scale), int(cur_height*scale)), PIL.Image.ANTIALIAS)
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    del img
    return bio.getvalue()


def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def open_window(fig):
    layout1 = [[sg.Text("Image Histogram", key="new")],
              [sg.Canvas(key='c')],
              [sg.Button("OK", key='-HIST OK-')]]

    win = sg.Window("Second Window", layout1, finalize=True)
    choice = None
    print("yes", win['c'])
    print(dir(win['c']))
    print('lp', win['c'].TKCanvas)
    draw_figure(win['c'].TKCanvas, fig)
    while True:
        event, values = win.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == '-HIST OK-':
            break

    win.close()


def make_window(theme):
    sg.theme(theme)
    menu_def = [['File', ['Open', 'Save', 'Save As', '---', 'Exit']],
                # ['Tools', ['Show', 'Hide']],
                # ['Reset', ['Reset All']],
                ['Help', ['About']]]
    right_click_menu_def = [[], ['Nothing', 'More Nothing', 'Exit']]

    # Table Data
    data = [["John", 10], ["Jen", 5]]
    headings = ["Name", "Score"]
    version = str(1.0)

    left_col = [
                [sg.Text('Folder'), sg.In(size=(15, 1), enable_events=True, key='-FOLDER-'), sg.FolderBrowse()],
                [sg.Text('File:   '),
                 sg.In(enable_events=True,  size=(15, 1), key='-IN FILE-'),
                 sg.FileBrowse()],
                [sg.Listbox(values=[], enable_events=True, expand_y=True, size=(30, 5), key='-FILE LIST-')],
                [sg.Text('Version ' + version, font='Courier 8')]]

    images_display_block = [
        [sg.Graph(canvas_size=(1200, 1500), graph_bottom_left=(0, 1500), graph_top_right=(1200, 0), pad=(0, 0),
                  background_color='white', key='graph', enable_events=True, tooltip='Raw dr images will display ')]
    ]
    basics_block = [
        [sg.Button('Basic Info', key='-BASIC INFO-', size=(30, 5), pad=(25, 2), tooltip='Display Basic Image Info!'),
        sg.Button('Histogram', key='-HISTOGRAM-', size=(30, 5), pad=(25, 2), tooltip='Display Image Histogram!'),
         ],
    ]
    scale_block = [
           [
            sg.Text('      20%', justification='center'),
            sg.Slider(orientation='h', default_value=1.0, range=(float(0.2), float(5)), key='-Scaling Ratio-',
                 enable_events=True, resolution=0.1, size=(20, 10), tooltip='Scaling Ratio',
                      # background_color='#1B2838'
                      ),
            sg.Text('500%', justification='center'),
           ]
      ]
    rotation_block = [
        [
            sg.Text('      0', justification='center'),
            sg.Slider(orientation='h', default_value=0, range=(0, 360), key='-Rotation Block-',
                      enable_events=True, resolution=1, size=(30, 10), tooltip='Rotation Degree',
                      # background_color='#1B2838'
                      ),
            sg.Text('360', justification='center'),
        ]
    ]
    graylevelwindow_block = [
        [
            sg.Text('  WindowLevel', justification='center', key='-level text-'),
            sg.Text('1', justification='center', key='-level start-'),
            sg.Slider(orientation='h', default_value=1, range=(1, 4094), key='-Window Level-',
                      enable_events=True, resolution=1, size=(30, 10), tooltip='Window Level',
                      # background_color='#1B2838'
                      ),
            sg.Text('4094', justification='center', key='-level end-'),
            sg.Text('  WindowWidth', justification='center'),
            sg.Text('2', justification='center', key='-width start-'),
            sg.Slider(orientation='h', default_value=2, range=(2, 4094), key='-Window Width-',
                      enable_events=True, resolution=2, size=(30, 10), tooltip='Window Width',
                      # background_color='#1B2838'
                      ),
            sg.Text('4094', justification='center', key='-width end-'),
        ]
    ]

    enhancement_block = [
        [sg.Button('HomomorphicFilter', key='-homomorphic_filter-', size=(30, 5),
                   pad=(25, 2), tooltip='Detail Enhancement!'),
         ],
    ]

    reset_block = [
        [sg.Button('Reset All', key='Reset All', size=(30, 5), pad=(25, 2),
                   tooltip='Reset All!'),
         ],
    ]

    # ----- Full layout -----
    layout = [ [sg.Menu(menu_def, key='-MENU-', font='Any 10')],
               [

               sg.Column(left_col, expand_y=True),
               sg.VSeperator(),
               sg.Column([
                   [
                     sg.Column([[sg.TabGroup([[
                        sg.Tab('                Basics              ', basics_block),
                        sg.Tab('            Scaling             ', scale_block, key='-scaling-'),
                        sg.Tab('                Rotation                ', rotation_block,),
                        sg.Tab('                GrayLevelWindow             ', graylevelwindow_block, key='-window-'),
                        sg.Tab('                Enhancement               ', enhancement_block),
                         sg.Tab('               Reset',  reset_block)
                    ]], size=(toolbar_width, toolbar_height), key='-Tabs-', )]],  key='-TAB GROUP-', visible=True, ),
                ],
                   [sg.Column(images_display_block, expand_x=True, expand_y=True, key='display', pad=(0, 10),
                              scrollable=True)]],
                   vertical_alignment='top', element_justification='left', key='cols', scrollable=False
               )
              ]
            ]

    return sg.Window('DR Image Viewer', layout, right_click_menu=right_click_menu_def,
                     resizable=True, grab_anywhere=True, no_titlebar=False), layout


def main():
    # window = make_window(sg.theme())
    global img_key, cur_img_arr, original_img_arr, cur_img_arr_normalized, original_img_name, temp_original_img_arr, temp_original_img_arr_normalized, pre_degree, original_img_arr_normalized
    print(sg.theme_list())
    window, layout = make_window('LightGrey1')
    # window, layout = make_window('Python')

    # This is an Event Loop
    while True:
        event, values = window.read()
        if event in (None, 'Exit'):
            break
        if event == '-FOLDER-':
            folder = values['-FOLDER-']
            allowed_img_types = ('raw', )
            try:
                img_lists = os.listdir(folder)
                if len(img_lists) == 0:
                    sg.popup('Prompts', 'Empty Folder!',
                             grab_anywhere=True)
            except Exception as e:
                print(e)
                continue

            fnames = [f for f in img_lists if os.path.isfile(os.path.join(folder, f)) and
                      f.endswith(allowed_img_types)]
            if len(fnames) == 0:
                sg.popup('Prompts',
                         'No raw files in this directory!',
                         grab_anywhere=True)
                window['-FOLDER-'].update('')
                continue
            window['-IN FILE-'].update('')
            window['-FILE LIST-'].update(fnames)
        elif event == '-IN FILE-':
            filename = values['-IN FILE-']
            if not filename.endswith(('raw', )):
                sg.popup('Prompts',
                         'Not a raw file!')
                window['-IN FILE-'].update('')
                continue
            s_time = time.time()
            try:
                original_img_name = filename
                window['-Rotation Block-'].update(value=0)
                window['-Scaling Ratio-'].update(value=1.0)

                original_img_arr = tools.read_raw_dr_img(filename)
                original_img_arr_normalized = tools.normalize_img(original_img_arr)
                print('read time', time.time() - s_time)

                imgbytes_in = cv2.imencode('.png', original_img_arr_normalized)[1].tobytes()
                print('trans time', time.time() - s_time)
                window['-FOLDER-'].update('')
                min_gray = np.min(original_img_arr)
                max_gray = np.max(original_img_arr)
                print('recompute the window level and window width info:', min_gray, ":", max_gray)
                window['-Window Level-'].Update(range=(min_gray+1, max_gray-1))
                mid_level = (min_gray + max_gray) // 2
                window['-Window Level-'].Update(value=min_gray+1)
                window['-level start-'].update(value=min_gray+1)
                window['-level end-'].update(value=max_gray-1)
                max_win_width = min(mid_level - min_gray, max_gray - mid_level) * 2
                window['-Window Width-'].Update(range=(2, max_win_width))
                window['-Window Width-'].update(value=2)
                window['-width start-'].update(value=2)
                window['-width end-'].update(value=2)

                if os.path.basename(filename) not in values['-FILE LIST-']:
                    l = list(values['-FILE LIST-'])
                    print('l is', l)
                    l.append(os.path.basename(filename))
                    print('l2 is', l)
                    window['-FILE LIST-'].update(l)

                if img_key is not None:
                    window['graph'].delete_figure(img_key)

                img_key = window['graph'].draw_image(data=imgbytes_in, location=(0, 0))
                window['graph'].Widget.configure(width=original_img_arr_normalized.shape[-1], height=original_img_arr_normalized.shape[0])
                max_width = max(700, original_img_arr_normalized.shape[-1])
                max_height = max(700, original_img_arr_normalized.shape[0])
                canvas = window['display'].Widget.canvas

                cur_img_arr_normalized = original_img_arr_normalized.copy()
                cur_img_arr = original_img_arr.copy()

            except Exception as e:
                print(e)
                continue
        elif event == '-FILE LIST-':
            try:
                filename = os.path.join(values['-FOLDER-'], values['-FILE LIST-'][0])
                window['-Rotation Block-'].update(value=0)
                window['-Scaling Ratio-'].update(value=1.0)
                window['-Window Level-'].update(value=1000)
                window['-Window Width-'].update(value=2)
                original_img_name = filename
                original_img_arr = tools.read_raw_dr_img(filename)
                original_img_arr_normalized = tools.normalize_img(original_img_arr)
                imgbytes_in = cv2.imencode('.png', cur_img_arr_normalized)[1].tobytes()

                min_gray = np.min(original_img_arr)
                max_gray = np.max(original_img_arr)
                print('recompute the window level and window width info:', min_gray, ":", max_gray)
                window['-Window Level-'].Update(range=(min_gray+1, max_gray-1))
                mid_level = (min_gray + max_gray) // 2
                window['-Window Level-'].Update(value=min_gray+1)
                window['-level start-'].update(value=min_gray+1)
                window['-level end-'].update(value=max_gray-1)
                max_win_width = min(mid_level - min_gray, max_gray - mid_level) * 2
                window['-Window Width-'].Update(range=(2, max_win_width))
                window['-Window Width-'].update(value=2)
                window['-width start-'].update(value=2)
                window['-width end-'].update(value=2)

                if img_key is not None:
                    window['graph'].delete_figure(img_key)
                img_key = window['graph'].draw_image(data=imgbytes_in, location=(0, 0))
                window['graph'].Widget.configure(width=original_img_arr_normalized.shape[-1],
                                                 height=original_img_arr_normalized.shape[0])
                max_width = max(700, original_img_arr_normalized.shape[-1])
                max_height = max(700, original_img_arr_normalized.shape[0])
                canvas = window['display'].Widget.canvas
                canvas.configure(scrollregion=(0, 0, max_width, max_height))

                cur_img_arr_normalized = original_img_arr_normalized.copy()
                cur_img_arr = original_img_arr.copy()

            except Exception as E:
                print(f'** Error {E} **')
                pass
        elif event == 'Show':
            print('show is happening')
            window['-TAB GROUP-'].update(visible=True)
            layout[1][2].expand()
        elif event == 'Hide':
            print('Hide is happening')
            window['-TAB GROUP-'].update(visible=False)
        elif event == '-BASIC INFO-':
            sg.popup('  Image Basic Info    ',
                      'image width:    ' + str(cur_img_arr.shape[-1]),
                      'image height:   ' + str(cur_img_arr.shape[0]),
                      'min gray level:'  + str(np.min(cur_img_arr)),
                      'max gary level:'  + str(np.max(cur_img_arr)), line_width=80)
            pass
        elif event == '-HISTOGRAM-':
            fig = tools.gray_histogram(cur_img_arr)
            open_window(fig)
        elif event == '-Scaling Ratio-':
            print('scale ratio is', values['-Scaling Ratio-'])
            scale_ratio = values['-Scaling Ratio-']

            if original_img_arr.shape == (1, 1):
                # sg.popup('no file now!')
                continue

            if last_event != event:
                temp_original_img_arr = cur_img_arr
                temp_original_img_arr_normalized = cur_img_arr_normalized
            new_h = int(original_img_arr.shape[0] * scale_ratio)
            new_w = int(original_img_arr.shape[1] * scale_ratio)
            print('new h', new_h)
            print('new w', new_w)
            cur_img_arr_normalized = cv2.resize(temp_original_img_arr_normalized, (new_h, new_w))
            print('+'*100)
            print('before resize', np.max(cur_img_arr))
            print(cur_img_arr.dtype)
            cur_img_arr = cv2.resize(cur_img_arr, (new_h, new_w))
            print('after resize', np.max(cur_img_arr))
            print('*'*100)

            if img_key is not None:
                window['graph'].delete_figure(img_key)

            imgbytes_in = cv2.imencode('.png', cur_img_arr_normalized)[1].tobytes()
            img_key = window['graph'].draw_image(data=imgbytes_in, location=(0, 0))
            window['graph'].Widget.configure(width=cur_img_arr_normalized.shape[-1],
                                             height=cur_img_arr_normalized.shape[0])
            max_width = max(700, cur_img_arr_normalized.shape[-1])
            max_height = max(700, cur_img_arr_normalized.shape[0])
            canvas = window['display'].Widget.canvas
            canvas.configure(scrollregion=(0, 0, max_width, max_height))

        elif event == '-Window Level-':
            if original_img_arr.shape == (1, 1):
                # sg.popup('no file now!')
                continue

            if last_event != event and last_event !='-Window Width-':
                print('----------------yes-----------------')
                print('last event:', last_event)
                print('event:', event)
                temp_original_img_arr = cur_img_arr
                print('cur shape:', cur_img_arr.shape)
                temp_original_img_arr_normalized = cur_img_arr_normalized

            cur_win_level = values['-Window Level-']
            min_gray = np.min(temp_original_img_arr)
            max_gray = np.max(temp_original_img_arr)
            print(f'min:{min_gray} max:{max_gray}')
            max_win_width = min(cur_win_level - min_gray, max_gray - cur_win_level) * 2
            print('max win width is:', max_win_width)
            window['-Window Width-'].Update(range=(2, max_win_width))
            window['-width end-'].update(value=max_win_width)
            window['-Window Width-'].update(value=max_win_width)
            cur_win_width = values['-Window Width-']

            cur_img_arr_normalized = tools.gray_level_transformation(temp_original_img_arr, cur_win_level, cur_win_width)
            # print('cur img arr is', cur_img_arr)
            cur_img_arr = tools.unnormalize_img(cur_img_arr_normalized)

            if img_key is not None:
                window['graph'].delete_figure(img_key)

            imgbytes_in = cv2.imencode('.png', cur_img_arr_normalized)[1].tobytes()
            img_key = window['graph'].draw_image(data=imgbytes_in, location=(0, 0))
            window['graph'].Widget.configure(width=cur_img_arr_normalized.shape[-1],
                                             height=cur_img_arr_normalized.shape[0])
            max_width = max(700, cur_img_arr_normalized.shape[-1])
            max_height = max(700, cur_img_arr_normalized.shape[0])
            canvas = window['display'].Widget.canvas
            canvas.configure(scrollregion=(0, 0, max_width, max_height))
        elif event == '-Window Width-':
            if original_img_arr.shape == (1, 1):
                # sg.popup('no file now!')
                continue

            if last_event != event and last_event != '-Window Level-':
                temp_original_img_arr = cur_img_arr
                temp_original_img_arr_normalized = cur_img_arr_normalized

            cur_win_level = values['-Window Level-']
            cur_win_width = values['-Window Width-']
            print('cur_win_width', cur_win_width)
            print('cur_win_level', cur_win_level)
            cur_img_arr_normalized = tools.gray_level_transformation(temp_original_img_arr, cur_win_level, cur_win_width)
            # print('cur img arr is', cur_img_arr)
            cur_img_arr = tools.unnormalize_img(cur_img_arr_normalized)

            if img_key is not None:
                window['graph'].delete_figure(img_key)

            imgbytes_in = cv2.imencode('.png', cur_img_arr_normalized)[1].tobytes()
            img_key = window['graph'].draw_image(data=imgbytes_in, location=(0, 0))
            window['graph'].Widget.configure(width=cur_img_arr_normalized.shape[-1],
                                             height=cur_img_arr_normalized.shape[0])
            max_width = max(700, cur_img_arr_normalized.shape[-1])
            max_height = max(700, cur_img_arr_normalized.shape[0])
            canvas = window['display'].Widget.canvas
            canvas.configure(scrollregion=(0, 0, max_width, max_height))
        elif event == 'Save':
            if original_img_name == '':
                continue
            original_img_arr = cur_img_arr
            original_img_arr_normalized = cur_img_arr_normalized
            saved_result = tools.save_raw_img(original_img_arr, original_img_name)
            if saved_result:
                sg.popup('  Result ', 'Image Saved!')
            else:
                sg.popup('  Result ', 'Errors occured!')
        elif event == 'Open':
            filename = sg.popup_get_file('请选择要打开的dicom文件', save_as=False, multiple_files=False, default_extension='.raw')

            if not filename.endswith(('raw', )):
                sg.popup('Prompts',
                         'Not a raw file!')
                window['-IN FILE-'].update('')
                continue
            s_time = time.time()
            try:
                original_img_name = filename
                original_img_arr = tools.read_raw_dr_img(filename)
                # original_img_arr = tools.read_raw_dr_img_faster(filename)
                original_img_arr_normalized = tools.normalize_img(original_img_arr)
                print('read time', time.time() - s_time)
                imgbytes_in = cv2.imencode('.png', original_img_arr_normalized)[1].tobytes()
                print('trans time', time.time() - s_time)
                window['-FOLDER-'].update('')

                if os.path.basename(filename) not in values['-FILE LIST-']:
                    l = list(values['-FILE LIST-'])
                    print('l is', l)
                    l.append(os.path.basename(filename))
                    print('l2 is', l)
                    window['-FILE LIST-'].update(l)

                if img_key is not None:
                    window['graph'].delete_figure(img_key)

                img_key = window['graph'].draw_image(data=imgbytes_in, location=(0, 0))
                window['graph'].Widget.configure(width=original_img_arr_normalized.shape[-1], height=original_img_arr_normalized.shape[0])
                max_width = max(700, original_img_arr_normalized.shape[-1])
                max_height = max(700, original_img_arr_normalized.shape[0])
                canvas = window['display'].Widget.canvas

                cur_img_arr_normalized = original_img_arr_normalized.copy()
                cur_img_arr = original_img_arr.copy()

            except Exception as e:
                print(e)
                continue
        elif event == 'Save As':
            if original_img_name == '':
                continue
            save_path = sg.popup_get_file('请选择文件另外保存的位置！', save_as=True, default_extension='.raw')
            saved_result = tools.save_raw_img(cur_img_arr, save_path)
            if saved_result:
                sg.popup('  Result ', 'Image Saved!')
            else:
                sg.popup('  Result ', 'Errors occured!')
        elif event == '-Rotation Block-':
            if original_img_name == '':
                continue
            if last_event != event:
                temp_original_img_arr = cur_img_arr
                temp_original_img_arr_normalized = cur_img_arr_normalized
                pre_degree = values['-Rotation Block-']

            rotation_degree = values['-Rotation Block-'] - pre_degree
            cur_img_arr_normalized = tools.rotate_bound(temp_original_img_arr_normalized, rotation_degree)
            cur_img_arr = tools.rotate_bound(temp_original_img_arr, rotation_degree)

            if img_key is not None:
                window['graph'].delete_figure(img_key)

            imgbytes_in = cv2.imencode('.png', cur_img_arr_normalized)[1].tobytes()
            img_key = window['graph'].draw_image(data=imgbytes_in, location=(0, 0))
            window['graph'].Widget.configure(width=cur_img_arr_normalized.shape[-1],
                                             height=cur_img_arr_normalized.shape[0])
            max_width = max(700, cur_img_arr_normalized.shape[-1])
            max_height = max(700, cur_img_arr_normalized.shape[0])
            canvas = window['display'].Widget.canvas
            canvas.configure(scrollregion=(0, 0, max_width, max_height))
        elif event == '-homomorphic_filter-':
            if original_img_name == '':
                continue
            if last_event != event:
                temp_original_img_arr = cur_img_arr
                temp_original_img_arr_normalized = cur_img_arr_normalized

            d0 = 17
            settings = dict(d0=d0, r1=0.5, rh=2, c=4, h=2.0, l=0.7)
            cur_img_arr_normalized = tools.homomorphic_filter(temp_original_img_arr, settings)

            cur_img_arr = tools.unnormalize_img(cur_img_arr_normalized)
            print('filtered:', np.max(cur_img_arr), ": dtype", cur_img_arr.dtype)

            if img_key is not None:
                window['graph'].delete_figure(img_key)

            imgbytes_in = cv2.imencode('.png', cur_img_arr_normalized)[1].tobytes()
            img_key = window['graph'].draw_image(data=imgbytes_in, location=(0, 0))
            window['graph'].Widget.configure(width=cur_img_arr_normalized.shape[-1],
                                             height=cur_img_arr_normalized.shape[0])
            max_width = max(700, cur_img_arr_normalized.shape[-1])
            max_height = max(700, cur_img_arr_normalized.shape[0])
            canvas = window['display'].Widget.canvas
            canvas.configure(scrollregion=(0, 0, max_width, max_height))


        elif event == '-histogram_equalization-':
            if original_img_name == '':
                continue
            if last_event != event:
                temp_original_img_arr = cur_img_arr
                temp_original_img_arr_normalized = cur_img_arr_normalized

            cur_img_arr_normalized = tools.gray_histogram_equalization(temp_original_img_arr)
            cur_img_arr = tools.unnormalize_img(cur_img_arr_normalized)

            if img_key is not None:
                window['graph'].delete_figure(img_key)

            imgbytes_in = cv2.imencode('.png', cur_img_arr_normalized)[1].tobytes()
            img_key = window['graph'].draw_image(data=imgbytes_in, location=(0, 0))
            window['graph'].Widget.configure(width=cur_img_arr_normalized.shape[-1],
                                             height=cur_img_arr_normalized.shape[0])
            max_width = max(700, cur_img_arr_normalized.shape[-1])
            max_height = max(700, cur_img_arr_normalized.shape[0])
            canvas = window['display'].Widget.canvas
            canvas.configure(scrollregion=(0, 0, max_width, max_height))
        elif event == 'Reset All':
            if original_img_name == '':
                continue

            window['-Rotation Block-'].update(value=0)
            window['-Scaling Ratio-'].update(value=1.0)

            imgbytes_in = cv2.imencode('.png', original_img_arr_normalized)[1].tobytes()
            min_gray = np.min(original_img_arr)
            max_gray = np.max(original_img_arr)
            print('recompute the window level and window width info:', min_gray, ":", max_gray)
            window['-Window Level-'].Update(range=(min_gray + 1, max_gray - 1))
            mid_level = (min_gray + max_gray) // 2
            window['-Window Level-'].Update(value=min_gray + 1)
            window['-level start-'].update(value=min_gray + 1)
            window['-level end-'].update(value=max_gray - 1)
            max_win_width = min(mid_level - min_gray, max_gray - mid_level) * 2
            window['-Window Width-'].Update(range=(2, max_win_width))
            window['-Window Width-'].update(value=2)
            window['-width start-'].update(value=2)
            window['-width end-'].update(value=2)

            cur_img_arr_normalized = original_img_arr_normalized
            cur_img_arr = original_img_arr

            if img_key is not None:
                window['graph'].delete_figure(img_key)

            imgbytes_in = cv2.imencode('.png', cur_img_arr_normalized)[1].tobytes()
            img_key = window['graph'].draw_image(data=imgbytes_in, location=(0, 0))
            window['graph'].Widget.configure(width=cur_img_arr_normalized.shape[-1],
                                             height=cur_img_arr_normalized.shape[0])
            max_width = max(700, cur_img_arr_normalized.shape[-1])
            max_height = max(700, cur_img_arr_normalized.shape[0])
            canvas = window['display'].Widget.canvas
            canvas.configure(scrollregion=(0, 0, max_width, max_height))
        else:
            pass
        last_event = event
    window.close()
    exit(0)


if __name__ == '__main__':
    main()

