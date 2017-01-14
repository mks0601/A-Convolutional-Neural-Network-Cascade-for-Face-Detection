import sqlite3

db_dir = "/media/sda1/Data/Face_detection/AFLW/aflw/data/aflw.sqlite"
save_dir = "/media/sda1/Data/Face_detection/AFLW/aflw/data/annot"

con = sqlite3.connect(db_dir)
cursor = con.cursor()

cursor.execute("select FaceImages.filepath, FaceRect.x, FaceRect.y, FaceRect.w, FaceRect.h from FaceImages join Faces on FaceImages.file_id = Faces.file_id join FaceRect on FaceRect.face_id = Faces.face_id")

result = cursor.fetchall()

#filename, x, y, w, h
fp = open(save_dir,"w")
for elem in result:
    print>>fp, elem


