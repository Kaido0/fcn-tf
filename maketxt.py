#制作训练label的txt，把像素点都存到txt中
def makelabeltxt():
    if not os.path.exists('data30ctxt/'):
        txtpath = os.mkdir('data30ctxt/')
    else:
        txtpath='data30ctxt/'
    data_path = 'data30c'
    images = os.listdir(data_path)
    count=0
    for image_name in images:
        if 'test' in image_name:
            continue
        print image_name
        count+=1
        imgname=image_name.split('.bmp')[0]
        image = os.path.join(data_path, image_name)
        img = np.array(Image.open(image).convert('L'))
        rows, cols = img.shape
        for i in range(rows):
            for j in range(cols):
                if (img[i,j] == 85):
                    img[i,j] = 1
                elif (img[i,j] == 170):
                    img[i,j] = 2
                elif (img[i,j] == 255):
                    img[i,j] = 3
                else:
                    img[i,j] = 0
        np.savetxt(txtpath+imgname+".txt", img, fmt="%d")

#保存训练数据的名字
def savetrainName():
    data_path='data30c'
    f=open('train.txt','w')
    images=os.listdir(data_path)
    for image_name in images:
        imname=image_name.split('.')[0]
        f.write(imname+'\n')

    f.close()
