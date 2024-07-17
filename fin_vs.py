import json
import re
import gradio as gr
import numpy as np
import pandas as pd
import os

class TextProcessor:
    def __init__(self, input_path, csv_path):
        self.input_path = input_path
        self.csv_path = csv_path
        self.original_lines, self.wav_files = self.load_original_files_and_wav_paths()
        self.total_lines = len(self.original_lines)
        self.current_choices = []
        self.current_choice_index = 0
        self.current_line_index = 0
        self.fi_cl_st = []
        self.current_line = ""
        self.csv_data = self.load_csv_data()
        self.load_progress()
        self.history = []
        self.previous_cleaned_text = ""
        self.previous_line = "처음입니다"
        self.all_cleaned_texts = [] 
        self.current_state_saved = False 
        self.line_numbers = []  

    def load_original_files_and_wav_paths(self):
        with open(self.input_path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
        txt_files = list(data.get('txt_files', {}).values())
        wav_files = list(data.get('wav_files', []))
        return txt_files, wav_files

    def load_csv_data(self):
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path, encoding='utf-8')
            return df.values.tolist()
        return []

    def save_to_csv(self):
        df = pd.DataFrame(self.csv_data, columns=['원본 텍스트', '수정된 텍스트', 'WAV 파일 경로'])
        df.to_csv(self.csv_path, index=False, encoding='utf-8')

    def load_progress(self):
        if self.csv_data:
            last_processed_line = self.csv_data[-1][0]
            if last_processed_line in self.original_lines:
                self.current_line_index = self.original_lines.index(last_processed_line) + 1

    def process_line(self, line):
        arr1, arr2, a, check = [], [], "", False

        j = 0
        while j < len(line):
            if line[j] == '\n':
                if arr2:
                    arr1.append(arr2)
                    arr2 = []
            elif line[j] == '(':
                check = True
            elif line[j] == ')':
                check = False
                if a:
                    if j + 2 < len(line) and line[j + 1:j + 3] == '/(':
                        arr3 = [a]
                        a = ""
                        j += 3
                        check = True
                        while j < len(line):
                            if line[j] == ')':
                                check = False
                                if a:
                                    arr3.append(a)
                                a = ""
                                break
                            elif check:
                                a += line[j]
                            j += 1
                        arr2.append(arr3)
                    else:
                        arr2.append([a, ""])
                a = ""
            elif check:
                a += line[j]
            j += 1

        if arr2:
            arr1.append(arr2)

        arr_np = np.array(arr1, dtype=object)
        cl_st = [item for sublist in arr_np for subsublist in sublist for item in subsublist]
        self.current_choices = cl_st
        self.current_choice_index = 0

    def construct_final_line(self, line, choices):
        new_line = ""
        l = 0
        choice_index = 0
        while l < len(line):
            if line[l] == '(':
                m = l
                while m < len(line) and line[m] != ')':
                    m += 1
                if m < len(line):
                    if m + 2 < len(line) and line[m + 1:m + 3] == '/(':
                        if choice_index < len(choices):
                            new_line += str(choices[choice_index])
                            choice_index += 1
                        else:
                            new_line += line[l:m+1]
                        m += 3
                        while m < len(line) and line[m] != ')':
                            m += 1
                        m += 1
                    else:
                        if choice_index < len(choices):
                            new_line += str(choices[choice_index])
                            choice_index += 1
                        else:
                            new_line += line[l:m+1]
                        m += 1
                l = m
            else:
                new_line += line[l]
                l += 1
        return new_line

    def clean_text(self, text):
        cleaned_text = re.sub(r'o/|n/|d/|\(\)|\)|\(|/', '', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return cleaned_text

    def get_next_line(self):
        while self.current_line_index < len(self.original_lines):
            self.current_line = self.original_lines[self.current_line_index]
            self.process_line(self.current_line)
            self.current_line_index += 1
            if len(self.current_choices) > 0:
                instructions = f"다음 중 하나를 선택하세요:\n1번: {self.current_choices[self.current_choice_index]}\n2번: {self.current_choices[self.current_choice_index + 1] if len(self.current_choices) > self.current_choice_index + 1 else '없음'}"
                self.history.append((self.current_line_index - 1, self.current_line, self.current_choices, self.current_choice_index, list(self.fi_cl_st), self.previous_cleaned_text))
                self.current_state_saved = True
                self.fi_cl_st = []  # Clear choices for each line
                position = f"{self.current_line_index} / {self.total_lines}"
                return self.current_line, instructions, "", self.previous_line, position
            else:
                final_text = self.clean_text(self.current_line)
                self.save_to_csv_data(self.current_line_index - 1, self.current_line, final_text, auto_processed=True)
                self.previous_cleaned_text = final_text
        return "모든 라인을 처리했습니다.", "", "", self.previous_line, f"{self.total_lines} / {self.total_lines}"

    def select_choice(self, choice):
        if self.current_choice_index < len(self.current_choices):
            self.fi_cl_st.append(choice)
            self.current_choice_index += 2
            current_partial_text = self.construct_final_line(self.current_line, self.fi_cl_st)
            cleaned_partial_text = self.clean_text(current_partial_text)
            if self.current_choice_index < len(self.current_choices):
                instructions = f"다음 중 하나를 선택하세요:\n1번: {self.current_choices[self.current_choice_index]}\n2번: {self.current_choices[self.current_choice_index + 1] if len(self.current_choices) > self.current_choice_index + 1 else '없음'}"
                position = f"{self.current_line_index} / {self.total_lines}"
                return self.current_line, instructions, cleaned_partial_text, self.previous_line, position
            else:
                final_line = self.construct_final_line(self.current_line, self.fi_cl_st)
                cleaned_text = self.clean_text(final_line)
                self.save_to_csv_data(self.current_line_index - 1, self.current_line, cleaned_text, auto_processed=False)
                self.previous_cleaned_text = cleaned_text
                self.previous_line = cleaned_text
                self.all_cleaned_texts.append(cleaned_text)  
                self.line_numbers.append(self.current_line_index - 1)  
                next_line, next_instructions, cleaned_text, previous_line, position = self.get_next_line()
                return next_line, next_instructions, cleaned_text, previous_line, position
        return "", "", "", self.previous_line, f"{self.current_line_index} / {self.total_lines}"

    def save_to_csv_data(self, line_index, original_text, cleaned_text, auto_processed):
        wav_path = self.wav_files[line_index] if line_index < len(self.wav_files) else ""
        if not auto_processed:
            self.previous_line = cleaned_text
            self.all_cleaned_texts.append(cleaned_text) 
            self.line_numbers.append(line_index) 
        existing_row_index = next((index for index, row in enumerate(self.csv_data) if row[0] == original_text), None)
        if existing_row_index is not None:
            self.csv_data[existing_row_index][1] = cleaned_text
            self.csv_data[existing_row_index][2] = wav_path
        else:
            self.csv_data.append([original_text, cleaned_text, wav_path])
        self.save_to_csv()

    def go_back(self):
        if self.history:
            last_state = self.history.pop()
            self.current_line_index, self.current_line, self.current_choices, self.current_choice_index, self.fi_cl_st, self.previous_cleaned_text = last_state
            instructions = f"다음 중 하나를 선택하세요:\n1번: {self.current_choices[self.current_choice_index]}\n2번: {self.current_choices[self.current_choice_index + 1] if len(self.current_choices) > self.current_choice_index + 1 else '없음'}"
            current_partial_text = self.construct_final_line(self.current_line, self.fi_cl_st)
            cleaned_partial_text = self.clean_text(current_partial_text)
            self.previous_line = self.previous_cleaned_text if self.history else "처음입니다"
            if self.all_cleaned_texts:
                self.previous_line = self.all_cleaned_texts.pop() 
            self.current_state_saved = False
            position = f"{self.current_line_index} / {self.total_lines}"
            return self.current_line, instructions, cleaned_partial_text, self.previous_line, position
        return "모든 라인을 처리했습니다.", "", "", "처음입니다", f"{self.total_lines} / {self.total_lines}"

    def go_to_last_saved(self):
        if self.line_numbers:
            self.current_line_index = self.line_numbers[-1] + 1
            return self.get_next_line()
        return "모든 라인을 처리했습니다.", "", "", "처음입니다", f"{self.total_lines} / {self.total_lines}"

input_path = "C:\\Users\\82109\\Desktop\\output_file.json"
csv_path = "C:\\Users\\82109\\Desktop\\output.csv"

processor = TextProcessor(input_path, csv_path)

def update_text(choice):
    next_line, instructions, cleaned_text, previous_cleaned_text, position = processor.select_choice(choice)
    if next_line == "모든 라인을 처리했습니다.":
        processor.save_to_csv()
    return next_line, instructions, cleaned_text, previous_cleaned_text, position

def get_initial_text():
    next_line, instructions, cleaned_text, previous_cleaned_text, position = processor.get_next_line()
    return next_line, instructions, cleaned_text, previous_cleaned_text, position

def go_back():
    next_line, instructions, cleaned_text, previous_cleaned_text, position = processor.go_back()
    return next_line, instructions, cleaned_text, previous_cleaned_text, position

def go_to_last_saved():
    next_line, instructions, cleaned_text, previous_cleaned_text, position = processor.go_to_last_saved()
    return next_line, instructions, cleaned_text, previous_cleaned_text, position

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("### 텍스트 선택기")
    input_text = gr.Textbox(lines=5, label="원본 텍스트", interactive=False)
    instruction_text = gr.Markdown()
    output_text = gr.Textbox(lines=5, label="수정된 텍스트", interactive=False)
    previous_output_text = gr.Textbox(lines=5, label="이전 수정된 텍스트", interactive=False)
    position_text = gr.Textbox(lines=1, label="현재 위치", interactive=False)

    start_button = gr.Button("처리 시작")
    start_button.click(get_initial_text, inputs=None, outputs=[input_text, instruction_text, output_text, previous_output_text, position_text])

    with gr.Row():
        button1 = gr.Button("1번 선택")
        button2 = gr.Button("2번 선택")
        back_button = gr.Button("뒤로가기")
        go_to_last_saved_button = gr.Button("마지막 저장 지점으로 이동")

    button1.click(lambda: update_text(processor.current_choices[processor.current_choice_index] if len(processor.current_choices) > processor.current_choice_index else ""), inputs=None, outputs=[input_text, instruction_text, output_text, previous_output_text, position_text])
    button2.click(lambda: update_text(processor.current_choices[processor.current_choice_index + 1] if len(processor.current_choices) > processor.current_choice_index + 1 else ""), inputs=None, outputs=[input_text, instruction_text, output_text, previous_output_text, position_text])
    back_button.click(go_back, inputs=None, outputs=[input_text, instruction_text, output_text, previous_output_text, position_text])
    go_to_last_saved_button.click(go_to_last_saved, inputs=None, outputs=[input_text, instruction_text, output_text, previous_output_text, position_text])

demo.launch()
