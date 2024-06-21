# Calculating Plant Height Using Computer Vision
## General Things To Know

1. app.py menggunakan Flask untuk server backend dan Roboflow untuk pretrained model Visi Komputer.
2. Folder templates berisikan file HTML. Untuk 'index.html' menggunakan fetch javascript untuk berkomunikasi dengan backend, dan 'index2.html' menggunakan axios.
3. Folder static dapat diisi file CSS dan JS.
4. File-file di folder test_image dapat digunakan untuk mengetes fitur upload website.
5. Hasil anotasi bounding box dan informasi tinggi tanaman akan diletakkan di folder uploads secara otomatis oleh program python.
6. Untuk menjalankan sistem ini, pertama jalankan Flask dengan command "python app.py". Kemudian webnya dapat dijalankan melalui live server vscode ataupun cara lainnya bebas.
   Perhatikan port dan host untuk Flask dan Website
