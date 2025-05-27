import tkinter as tk
from tkinter import filedialog, simpledialog
import cv2
from PIL import Image, ImageOps
import os
import numpy as np
import gc  # garbage collection
from reportlab.pdfgen import canvas
import traceback
import psutil  # dla monitorowania pamięci - pip install psutil

# ---------- USTAWIENIA ----------
DPI = 300  # Przyjmujemy 300 DPI
MAX_FRAME_SIZE = (1920, 1080)  # Maksymalny rozmiar klatki przed skalowaniem

def get_a4_dims(orient):
    """
    Zwraca (page_width, page_height) w pikselach dla kartki A4
    w orientacji "pionowa" (210x297 mm) lub "pozioma" (297x210 mm).
    """
    if orient == "pionowa":
        page_width = int(210 / 25.4 * DPI)
        page_height = int(297 / 25.4 * DPI)
    else:
        page_width = int(297 / 25.4 * DPI)
        page_height = int(210 / 25.4 * DPI)
    return page_width, page_height

# Marginesy: Górny margines – 20 mm, dolny pasek – 5 mm.
TOP_MARGIN = int(20 / 25.4 * DPI)
BOTTOM_MARGIN = int(5 / 25.4 * DPI)

# Format układu siatki kolażu zapisany jako (liczba kolumn, liczba wierszy)
GRID_FORMATS = [(2, 2), (3, 2), (3, 3), (2, 3), (4, 3)]
GRID_FORMATS_LABEL = {
    (2, 2): "2x2",
    (3, 2): "3x2",
    (3, 3): "3x3",
    (2, 3): "2x3",
    (4, 3): "4x3"
}

# Ustalamy metodę resamplingu – kompatybilną ze starszymi i nowszymi wersjami Pillow
try:
    resample_method = Image.Resampling.LANCZOS
except AttributeError:
    resample_method = Image.LANCZOS

def log_memory_usage(context=""):
    """Loguje aktualne zużycie pamięci"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"[PAMIĘĆ] {context}: {memory_mb:.1f} MB")

def resize_frame_if_needed(pil_img, max_size=MAX_FRAME_SIZE):
    """
    Zmniejsza rozmiar klatki jeśli przekracza maksymalne wymiary.
    """
    w, h = pil_img.size
    max_w, max_h = max_size

    if w > max_w or h > max_h:
        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        pil_img = pil_img.resize((new_w, new_h), resample=resample_method)
        print(f"Zmniejszono klatkę z {w}x{h} do {new_w}x{new_h}")

    return pil_img

def paste_image_in_cell(img, cell_w, cell_h):
    """
    Skaluje obraz (z zachowaniem proporcji) tak, aby zmieścił się w komórce o wymiarach cell_w x cell_h,
    a następnie wyśrodkowuje go na białym tle.
    """
    w, h = img.size
    scale = min(cell_w / w, cell_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = img.resize((new_w, new_h), resample=resample_method)
    cell_img = Image.new("RGB", (cell_w, cell_h), "white")
    offset_x = (cell_w - new_w) // 2
    offset_y = (cell_h - new_h) // 2
    cell_img.paste(resized, (offset_x, offset_y))
    return cell_img

def extract_single_frame(cap, frame_index, total_frames, orientation, frame_rotation):
    """
    Pobiera pojedynczą klatkę z filmu. Zwraca None w przypadku błędu.
    """
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        if not ret:
            # Szukamy najbliższej dostępnej klatki (zarówno wcześniejszej jak i późniejszej)
            for offset in range(1, min(50, total_frames)):
                if frame_index - offset >= 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index - offset)
                    ret, frame = cap.read()
                    if ret:
                        break
                if frame_index + offset < total_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index + offset)
                    ret, frame = cap.read()
                    if ret:
                        break

        if not ret:
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)

        pil_img = resize_frame_if_needed(pil_img)

        w, h = pil_img.size
        if frame_rotation != "Brak":
            if frame_rotation == "90° w prawo":
                pil_img = pil_img.rotate(-90, expand=True)
            elif frame_rotation == "90° w lewo":
                pil_img = pil_img.rotate(90, expand=True)
        else:
            # Automatyczna rotacja dla orientacji pionowej (jeśli obraz jest szerszy niż wyższy)
            if orientation == "pionowa" and w > h:
                pil_img = pil_img.rotate(90, expand=True)

        return pil_img

    except Exception as e:
        print(f"Błąd podczas pobierania klatki {frame_index}: {e}")
        return None

def create_collage_page_optimized(video_path, frame_indices, grid_cols, grid_rows, 
                                  page_width, page_height, orientation, frame_rotation):
    """
    Tworzy stronę kolażu pobierając klatki na bieżąco (bez przechowywania wszystkich w pamięci).
    """
    available_w = page_width
    available_h = page_height - TOP_MARGIN - BOTTOM_MARGIN
    cell_w = available_w // grid_cols
    cell_h = available_h // grid_rows

    collage = Image.new("RGB", (available_w, available_h), "white")
    total_cells = grid_cols * grid_rows

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Nie można otworzyć pliku: {video_path}")
        return collage

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    last_valid_frame = None

    try:
        for i in range(total_cells):
            col = i % grid_cols
            row = i // grid_cols

            if i < len(frame_indices):
                frame = extract_single_frame(cap, frame_indices[i], total_frames, orientation, frame_rotation)
                if frame is not None:
                    last_valid_frame = frame
            else:
                frame = None

            if frame is None and last_valid_frame is not None:
                frame = last_valid_frame

            if frame is not None:
                cell = paste_image_in_cell(frame, cell_w, cell_h)
            else:
                cell = Image.new("RGB", (cell_w, cell_h), "white")

            x = col * cell_w
            y = row * cell_h
            collage.paste(cell, (x, y))

            if frame is not None and frame is not last_valid_frame:
                del frame
            del cell

    finally:
        cap.release()
        if last_valid_frame is not None:
            del last_valid_frame

    return collage

def create_a4_page(collage, page_width, page_height):
    """
    Tworzy pełną stronę A4 z kolażem wklejonym od góry (z marginesem TOP_MARGIN).
    """
    page = Image.new("RGB", (page_width, page_height), "white")
    page.paste(collage, (0, TOP_MARGIN))
    return page

def generate_collage_pages_optimized(video_file, num_pages, grid, page_width, page_height, 
                                     orientation, frame_rotation):
    """
    Generuje kolaże (strony) z pliku wideo w sposób zoptymalizowany pod kątem pamięci.
    """
    cols, rows = grid
    cells_per_page = cols * rows
    total_frames_needed = num_pages * cells_per_page

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Nie można otworzyć pliku: {video_file}")
        return []

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total_video_frames <= 0:
        print(f"Brak klatek w pliku: {video_file}")
        return []

    if total_frames_needed > total_video_frames:
        total_frames_needed = total_video_frames

    frame_indices = [int(round(x)) for x in np.linspace(0, total_video_frames - 1, num=total_frames_needed)]

    pages = []
    print(f"Tworzenie {num_pages} stron kolażu dla pliku {video_file}...")
    log_memory_usage(f"Przed przetwarzaniem {os.path.basename(video_file)}")

    for page_num in range(num_pages):
        start_idx = page_num * cells_per_page
        end_idx = min(start_idx + cells_per_page, len(frame_indices))
        page_frame_indices = frame_indices[start_idx:end_idx]

        print(f"Tworzenie strony {page_num + 1}/{num_pages}...")

        collage = create_collage_page_optimized(
            video_file, page_frame_indices, cols, rows, 
            page_width, page_height, orientation, frame_rotation
        )

        full_page = create_a4_page(collage, page_width, page_height)
        pages.append(full_page)

        del collage
        gc.collect()

        log_memory_usage(f"Po stronie {page_num + 1}")

    print(f"Utworzono {len(pages)} stron kolażu dla pliku {video_file}.")
    return pages

def save_pages_to_pdf(pages, pdf_path, page_width, page_height):
    """
    Łączy strony kolażu w dokument PDF.
    """
    print(f"Tworzenie PDF: {pdf_path}")
    try:
        c = canvas.Canvas(pdf_path, pagesize=(page_width, page_height))
        for i, page in enumerate(pages):
            temp_path = f"temp_page_{i}.jpg"
            try:
                page.save(temp_path, "JPEG", quality=85)
                c.drawInlineImage(temp_path, 0, 0, width=page_width, height=page_height)
                c.showPage()
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        c.save()
        print(f"PDF zapisany: {pdf_path}")
    except Exception as e:
        print(f"Błąd podczas tworzenia PDF {pdf_path}: {e}")
        traceback.print_exc()

def get_user_options():
    """
    Wyświetla okno dialogowe opcji kolażu i pobiera wybory użytkownika.
    """
    options_window = tk.Toplevel()
    options_window.title("Opcje kolażu")
    # Nie ustawiamy sztywnej geometrii – okno zostanie dopasowane do zawartości.

    tk.Label(options_window, text="Wybierz orientację kartki:").pack(anchor="w", padx=10, pady=5)
    orientation_var = tk.StringVar(value="pozioma")  # Domyślnie "pozioma"
    tk.Radiobutton(options_window, text="Pionowa", variable=orientation_var, value="pionowa").pack(anchor="w", padx=20)
    tk.Radiobutton(options_window, text="Pozioma", variable=orientation_var, value="pozioma").pack(anchor="w", padx=20)
    
    tk.Label(options_window, text="Wybierz format(y) kolażu:").pack(anchor="w", padx=10, pady=5)
    grid_vars = {}
    for fmt in GRID_FORMATS:
        var = tk.BooleanVar(value=False)
        if fmt == (2, 2):
            var.set(True)
        tk.Checkbutton(options_window, text=GRID_FORMATS_LABEL[fmt], variable=var).pack(anchor="w", padx=20)
        grid_vars[fmt] = var

    tk.Label(options_window, text="Ilość stron kolażu na film:").pack(anchor="w", padx=10, pady=5)
    pages_var = tk.StringVar(value="1")
    tk.Entry(options_window, textvariable=pages_var, width=10).pack(anchor="w", padx=20)
    
    tk.Label(options_window, text="Rotacja klatek:").pack(anchor="w", padx=10, pady=5)
    rotation_var = tk.StringVar(value="Brak")
    tk.Radiobutton(options_window, text="Brak", variable=rotation_var, value="Brak").pack(anchor="w", padx=20)
    tk.Radiobutton(options_window, text="90° w prawo", variable=rotation_var, value="90° w prawo").pack(anchor="w", padx=20)
    tk.Radiobutton(options_window, text="90° w lewo", variable=rotation_var, value="90° w lewo").pack(anchor="w", padx=20)
    
    tk.Label(options_window, text="Wybierz format(y) wyjściowe:").pack(anchor="w", padx=10, pady=5)
    out_format_jpg_var = tk.BooleanVar(value=True)
    out_format_pdf_var = tk.BooleanVar(value=False)
    tk.Checkbutton(options_window, text="JPG", variable=out_format_jpg_var).pack(anchor="w", padx=20)
    tk.Checkbutton(options_window, text="PDF", variable=out_format_pdf_var).pack(anchor="w", padx=20)
    
    done = tk.BooleanVar(value=False)
    def finish():
        done.set(True)
        options_window.destroy()
    
    tk.Button(options_window, text="START", command=finish, bg="green", fg="white").pack(pady=20)
    
    options_window.update_idletasks()
    options_window.geometry(f"{options_window.winfo_reqwidth()}x{options_window.winfo_reqheight()}")
    
    options_window.transient()
    options_window.grab_set()
    options_window.wait_variable(done)
    
    selected_formats = [fmt for fmt, var in grid_vars.items() if var.get()]
    try:
        num_pages = int(pages_var.get())
        if num_pages < 1:
            num_pages = 1
    except ValueError:
        num_pages = 1

    print("Opcje użytkownika:")
    print("  Orientacja:", orientation_var.get())
    print("  Format(y) kolażu:", [GRID_FORMATS_LABEL[fmt] for fmt in selected_formats])
    print("  Ilość stron:", num_pages)
    print("  Rotacja:", rotation_var.get())
    print("  Format(y) wyjściowe:", "JPG" if out_format_jpg_var.get() else "", "PDF" if out_format_pdf_var.get() else "")
    
    return (orientation_var.get(), selected_formats, num_pages, 
            rotation_var.get(), out_format_jpg_var.get(), out_format_pdf_var.get())

def create_new_name_map(file_paths):
    """
    Dla listy plików tworzy mapowanie oryginalna nazwa -> nowa standaryzowana nazwa.
    """
    name_counter = {}
    new_names = {}
    for f in sorted(file_paths):
        base = os.path.basename(f)
        prefix = base[:4]
        if prefix in name_counter:
            name_counter[prefix] += 1
        else:
            name_counter[prefix] = 1
        new_base = f"{prefix}_{name_counter[prefix]}"
        new_names[f] = new_base
    return new_names

if __name__ == "__main__":
    try:
        root = tk.Tk()
        root.withdraw()
        
        print("=== GENERATOR KOLAŻU DAV ===")
        log_memory_usage("Start programu")
        
        print("Wybierz pliki wideo...")
        file_paths = filedialog.askopenfilenames(
            title="Wybierz pliki wideo", 
            filetypes=[("Pliki wideo", "*.dav *.mp4 *.avi *.mkv *.mov *.flv"), ("Wszystkie pliki", "*.*")]
        )
        print("Wybrane pliki:", file_paths)
        if not file_paths:
            print("Nie wybrano plików. Program kończy działanie.")
            exit(0)
        
        print("Wybierz katalog wynikowy...")
        out_dir = filedialog.askdirectory(title="Wybierz katalog dla wyników")
        if not out_dir:
            out_dir = os.path.join(os.path.dirname(file_paths[0]), "WYNIKI")
        else:
            out_dir = os.path.join(out_dir, "WYNIKI")
        os.makedirs(out_dir, exist_ok=True)
        print("Katalog wynikowy:", out_dir)
        
        (orientation, selected_formats, num_pages,
         frame_rotation, out_format_jpg, out_format_pdf) = get_user_options()
        page_width, page_height = get_a4_dims(orientation)
        print(f"Wymiary strony A4: {page_width} x {page_height}")
        
        new_name_map = create_new_name_map(file_paths)
        print("Mapowanie nowych nazw plików:", new_name_map)
        
        raport_lines = []
        total_files = len(file_paths)
        
        for fmt_idx, fmt in enumerate(selected_formats):
            fmt_label = GRID_FORMATS_LABEL[fmt]
            fmt_dir = os.path.join(out_dir, fmt_label)
            if out_format_jpg:
                jpg_dir = os.path.join(fmt_dir, "jpg")
                os.makedirs(jpg_dir, exist_ok=True)
            if out_format_pdf:
                pdf_dir = os.path.join(fmt_dir, "pdf")
                os.makedirs(pdf_dir, exist_ok=True)
            print(f"\n{'='*60}")
            print(f"Przetwarzanie formatu: {fmt_label} ({fmt_idx+1}/{len(selected_formats)})")
            print(f"{'='*60}")
            
            for file_idx, video_file in enumerate(sorted(file_paths)):
                try:
                    new_base = new_name_map[video_file]
                    if fmt_idx == 0:
                        raport_lines.append(f"Oryginal: {os.path.basename(video_file)}  ->  Nowa nazwa: {new_base}")
                    
                    print(f"\n[{file_idx+1}/{total_files}] Przetwarzam plik: {os.path.basename(video_file)}")
                    log_memory_usage(f"Przed plikiem {file_idx+1}")
                    
                    pages = generate_collage_pages_optimized(
                        video_file, num_pages, fmt, page_width, page_height, 
                        orientation, frame_rotation
                    )
                    
                    if not pages:
                        print(f"Brak klatek dla pliku {video_file}. Pomijam plik.")
                        continue
                    
                    if out_format_jpg:
                        for i, page in enumerate(pages, start=1):
                            jpg_filename = os.path.join(jpg_dir, f"{new_base}_{fmt_label}_strona{i}.jpg")
                            page.save(jpg_filename, "JPEG", quality=85)
                            print(f"Zapisano JPEG: {os.path.basename(jpg_filename)}")
                    
                    if out_format_pdf:
                        pdf_filename = os.path.join(pdf_dir, f"{new_base}_{fmt_label}.pdf")
                        save_pages_to_pdf(pages, pdf_filename, page_width, page_height)
                    
                    del pages
                    gc.collect()
                    log_memory_usage(f"Po pliku {file_idx+1}")
                    
                    print(f"✓ Zakończono przetwarzanie: {os.path.basename(video_file)}")
                    
                except Exception as e:
                    print(f"❌ BŁĄD podczas przetwarzania pliku {video_file}:")
                    print(f"   {e}")
                    traceback.print_exc()
                    gc.collect()
        
        raport_path = os.path.join(out_dir, "raport.txt")
        with open(raport_path, "w", encoding="utf-8") as raport_file:
            raport_file.write("RAPORT GENEROWANIA KOLAŻU\n")
            raport_file.write("="*50 + "\n\n")
            raport_file.write(f"Data: {os.path.basename(__file__)}\n")
            raport_file.write(f"Orientacja: {orientation}\n")
            raport_file.write(f"Formaty: {[GRID_FORMATS_LABEL[fmt] for fmt in selected_formats]}\n")
            raport_file.write(f"Strony na film: {num_pages}\n")
            raport_file.write(f"Rotacja: {frame_rotation}\n")
            raport_file.write(f"Formaty wyjściowe: {'JPG' if out_format_jpg else ''} {'PDF' if out_format_pdf else ''}\n\n")
            raport_file.write("MAPOWANIE NAZW:\n")
            raport_file.write("-" * 30 + "\n")
            raport_file.write("\n".join(raport_lines))
        print(f"\n✓ Utworzono raport: {raport_path}")
        
        print("\n" + "="*60)
        print("🎉 GENEROWANIE KOLAŻU ZAKOŃCZONE POMYŚLNIE!")
        print("="*60)
        log_memory_usage("Koniec programu")
        
    except Exception as e:
        print(f"\n❌ KRYTYCZNY BŁĄD PROGRAMU:")
        print(f"   {e}")
        traceback.print_exc()
    
    finally:
        gc.collect()
    
    input("\nNaciśnij Enter, aby zakończyć...")
