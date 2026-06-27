import sys
import tkinter as tk


class TkModeSelector:
    def __init__(self, root):
        self.root = root
        self.result = None
        
        self.root.title('Mode Selection')
        self.root.geometry('380x280')
        self.root.resizable(False, False)
        
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (380 // 2)
        y = (self.root.winfo_screenheight() // 2) - (280 // 2)
        self.root.geometry(f'380x280+{x}+{y}')
        
        self.root.configure(bg='#FFFFFF')
        
        title = tk.Label(
            root, 
            text='Select Mode',
            font=('Segoe UI', 14, 'bold'),
            bg='#FFFFFF',
            fg='#2C3E50'
        )
        title.pack(pady=(30, 25))
        
        button_frame = tk.Frame(root, bg='#FFFFFF')
        button_frame.pack(pady=5, padx=40, fill='both', expand=True)
        
        btn_full = tk.Button(
            button_frame,
            text='FULL',
            command=lambda: self.select_mode('FULL'),
            font=('Segoe UI', 13, 'bold'),
            bg='#3498DB',
            fg='white',
            relief='flat',
            cursor='hand2',
            activebackground='#2980B9',
            activeforeground='white',
            bd=0,
            height=2
        )
        btn_full.pack(fill='x', pady=6)
        
        btn_distance = tk.Button(
            button_frame,
            text='DISTANCE',
            command=lambda: self.select_mode('DISTANCE'),
            font=('Segoe UI', 13, 'bold'),
            bg='#E67E22',
            fg='white',
            relief='flat',
            cursor='hand2',
            activebackground='#D35400',
            activeforeground='white',
            bd=0,
            height=2
        )
        btn_distance.pack(fill='x', pady=6)
        
        btn_velocity = tk.Button(
            button_frame,
            text='VELOCITY',
            command=lambda: self.select_mode('VELOCITY'),
            font=('Segoe UI', 13, 'bold'),
            bg='#27AE60',
            fg='white',
            relief='flat',
            cursor='hand2',
            activebackground='#229954',
            activeforeground='white',
            bd=0,
            height=2
        )
        btn_velocity.pack(fill='x', pady=6)
        
        self.root.protocol("WM_DELETE_WINDOW", lambda: self.select_mode('FULL'))
    
    def select_mode(self, mode):
        self.result = mode
        self.root.quit()
        self.root.destroy()
    
    def get_mode(self):
        return self.result if self.result else 'FULL'


def select_mode_gui():
    root = tk.Tk()
    dialog = TkModeSelector(root)
    root.mainloop()
    
    mode = dialog.get_mode()
    return mode if mode else 'FULL'


if __name__ == '__main__':
    mode = select_mode_gui()
    print(f"Selected mode: {mode}")
