#!/usr/bin/env python
# coding: utf-8

# In[19]:


# Import libraries
from matplotlib import pyplot as plt
from matplotlib.image import imread
from numpy.linalg import inv
import numpy as np
import os


# In[ ]:





# In[20]:


dataset_path = 'Archive2/train_s/'
dataset_dir  = os.listdir(dataset_path)

width  = 70
height = 80

    


# In[21]:


number_of_classes=40
img_in_class=6

print('Train Images:')

# to store all the training images in an array
training_tensor   = np.ndarray(shape=(number_of_classes*img_in_class, height*width), dtype=np.float64)
c=1
for i in range(number_of_classes):
    for j in range(img_in_class):
        img = plt.imread(dataset_path+str(c)+'_'+str(i+1)+'.jpg')
        # copying images to the training array
        training_tensor[img_in_class*i+j,:] = np.array(img, dtype='float64').flatten()
        # plotting the training images
        plt.subplot(number_of_classes,img_in_class,1+img_in_class*i+j)
        plt.imshow(img, cmap='gray')
        plt.subplots_adjust(right=1.2, top=2.5)
        plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
        c=c+1
    c=c+4
plt.show()


# In[66]:


dataset_path1 = 'Archive2/test_s/'

#dataset_dir  = os.listdir(dataset_path)
print('Test Images:')
testing_tensor = np.ndarray(shape=(164, height*width), dtype=np.float64)
c2=1
for i in range(164):
    img = imread(dataset_path1+'/'+str(i+1)+'.jpg')
    testing_tensor[i,:] = np.array(img, dtype='float64').flatten()
    plt.subplot(41,4,1+i)
    plt.imshow(img, cmap='gray')
    plt.subplots_adjust(right=1.2, top=2.5)
    plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
plt.show()


# In[67]:


def PCA(training_tensor, number_chosen_components):
    
    mean_face = np.zeros((1,height*width))
    for i in training_tensor:
        mean_face = np.add(mean_face,i)
    mean_face = np.divide(mean_face,float(training_tensor.shape[0])).flatten()
    
#     plt.title('Mean face')
#     plt.imshow(mean_face.reshape(height, width), cmap='gray')
#     plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
#     plt.show()
    
#     print('Normalized faces:')
    normalised_training_tensor = np.ndarray(shape=(training_tensor.shape))
    for i in range(training_tensor.shape[0]):
        normalised_training_tensor[i] = np.subtract(training_tensor[i],mean_face)        
#     for i in range(len(training_tensor)):
#         img = normalised_training_tensor[i].reshape(height,width)
#         plt.subplot(10,6,1+i)
#         plt.imshow(img, cmap='gray')
#         plt.subplots_adjust(right=1.2, top=2.5)
#         plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
#     plt.show()
    cov_matrix = np.cov(normalised_training_tensor)
    cov_matrix = np.divide(cov_matrix,float(training_tensor.shape[0]))
 
    eigenvalues, eigenvectors, = np.linalg.eig(cov_matrix)
    eig_pairs = [(eigenvalues[index], eigenvectors[:,index]) for index in range(len(eigenvalues))]

    # Sort the eigen pairs in descending order:
    eig_pairs.sort(reverse=True)
    eigvalues_sort  = [eig_pairs[index][0] for index in range(len(eigenvalues))]
    eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]
    
    reduced_data = np.array(eigvectors_sort[:number_chosen_components]).transpose()

    proj_data = np.dot(training_tensor.transpose(),reduced_data)
    proj_data = proj_data.transpose()
    
#     print('Projected Data:')
#     for i in range(proj_data.shape[0]):
#         img = proj_data[i].reshape(height,width)
#         plt.subplot(10,10,1+i)
#         plt.imshow(img, cmap='jet')
#         plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
#         plt.subplots_adjust(right=1.2, top=2.5)
#     plt.show()

    wx = np.array([np.dot(proj_data,img) for img in normalised_training_tensor])
    
    return proj_data, wx


# In[68]:


# get the projected faces
number_chosen_components = 30
projected_data, projected_sig = PCA(training_tensor, number_chosen_components)
projected_sig.shape


# In[69]:


mew = np.zeros((number_of_classes, number_chosen_components))
M = np.zeros((1,number_chosen_components))
#print(mew)
#print(M)

for i in range(number_of_classes):
    xa = projected_sig[img_in_class*i:img_in_class*i+img_in_class,:]
    #print(xa)
    for j in xa:
        mew[i,:] = np.add(mew[i,:],j)
    mew[i,:] = np.divide(mew[i,:],float(len(xa)))

for i in projected_sig:
    M = np.add(M,i)
M = np.divide(M,float(len(projected_sig)))

print(M)
print(mew)

M.shape
mew.shape


# In[70]:


# normalised within class data
normalised_wc_proj_sig = np.ndarray(shape=(number_of_classes*img_in_class, number_chosen_components), dtype=np.float64)
#print(normalised_wc_proj_sig)
print(projected_sig)
print(mew)
for i in range(number_of_classes):
    for j in range(img_in_class):
        normalised_wc_proj_sig[i*img_in_class+j,:] = np.subtract(projected_sig[i*img_in_class+j,:],mew[i,:])
normalised_wc_proj_sig.shape

sw = np.zeros((number_chosen_components,number_chosen_components))
#print(normalised_wc_proj_sig)
for i in range(number_of_classes):
    xa = normalised_wc_proj_sig[img_in_class*i:img_in_class*i+img_in_class,:]
    xa = xa.transpose()
    cov = np.dot(xa,xa.T)
    sw = sw + cov
    #print(sw)
sw.shape


# In[71]:


normalised_proj_sig = np.ndarray(shape=(number_of_classes*img_in_class, number_chosen_components), dtype=np.float64)
for i in range(number_of_classes*img_in_class):
    normalised_proj_sig[i,:] = np.subtract(projected_sig[i,:],M)

sb = np.dot(normalised_proj_sig.T,normalised_proj_sig)
sb = np.multiply(sb,float(img_in_class))
sb.shape


# In[72]:


J = np.dot(inv(sw), sb)
J.shape


# In[73]:


eigenvalues, eigenvectors, = np.linalg.eig(J)
# eigenvectors = abs(eigenvectors)
print('Eigenvectors of Cov(X):')
print(eigenvectors)
# eigenvalues = abs(eigenvalues)
print('Eigenvalues of Cov(X):',eigenvalues)


# In[74]:


# get corresponding eigenvectors to eigen values
# so as to get the eigenvectors at the same corresponding index to eigen values when sorted
eig_pairs = [(eigenvalues[index], eigenvectors[:,index]) for index in range(len(eigenvalues))]

# Sort the eigen pairs in descending order:
eig_pairs.sort(reverse=True)
eigvalues_sort  = [eig_pairs[index][0] for index in range(len(eigenvalues))]
eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]

# Find cumulative variance of each principle component
var_comp_sum = np.cumsum(eigvalues_sort)/sum(eigvalues_sort)

# Show cumulative proportion of varaince with respect to components
print("Cumulative proportion of variance explained vector:", var_comp_sum)

# x-axis for number of principal components kept
num_comp = range(1,len(eigvalues_sort)+1)
plt.title('Cum. Prop. Variance and Components Kept')
plt.xlabel('Principal Components')
plt.ylabel('Cum. Prop. Variance ')

plt.scatter(num_comp, var_comp_sum)
plt.show()


# In[75]:


print('Number of eigen vectors:',len(eigvalues_sort))

# Choosing the necessary number of principle components
number_chosen_components = 15
print("k:",number_chosen_components)
reduced_data = np.array(eigvectors_sort[:number_chosen_components]).transpose()
reduced_data.shape


# In[76]:


projected_sig.shape
FP = np.dot(projected_sig, reduced_data)
FP.shape


# In[77]:


# get projected data ---> eigen space

proj_data1 = np.dot(training_tensor.transpose(),FP)
proj_data1 = proj_data1.transpose()
proj_data1.shape

# plotting of eigen faces --> the information retained after applying lossing transformation
for i in range(proj_data1.shape[0]):
    img = proj_data1[i].reshape(height,width)
    #print(img)
    plt.subplot(10,3,1+i)
    plt.imshow((img.real*img.real + img.imag*img.imag)**0.5, cmap='gray')
    plt.subplots_adjust(right=1.2, top=2.5)
    plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
plt.show()


# In[78]:


mean_face = np.zeros((1,height*width))

for i in training_tensor:
    mean_face = np.add(mean_face,i)

mean_face = np.divide(mean_face,float(len(training_tensor))).flatten()

plt.imshow(mean_face.reshape(height, width), cmap='gray')
plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
plt.show()


# In[79]:



# Testing all the images



count=0
num_images=0
correct_pred=0

false_pos=0
false_neg=0
def recogniser(img_number):
    global count,highest_min,num_images,correct_pred,false_pos,false_neg
    
    num_images          += 1
    unknown_face_vector = testing_tensor[img_number,:]
    normalised_uface_vector = np.subtract(unknown_face_vector,mean_face)
    
    plt.subplot(41,8,1+count)
    plt.imshow(unknown_face_vector.reshape(height,width), cmap='gray')
    plt.title('Input:'+str(img_number+1), fontdict = {'fontsize' : 50})
    plt.subplots_adjust(left = 15, right=20, top=20) #plt.subplots_adjust(right=1.2, top=2.5)
    plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
    count+=1
    
    PEF = np.dot(projected_data,normalised_uface_vector)
    proj_fisher_test_img = np.dot(reduced_data.T,PEF)
    diff  = FP - proj_fisher_test_img
    norms = np.linalg.norm(diff, axis=1)
#     print(norms.shape)
    index = np.argmin(norms)
    
    plt.subplot(41,8,1+count)
    
    set_number = int(img_number/4)
#     print(set_number)

    t0 = 7000000
    
#     if(img_number>=40):
#         print(norms[index])
    
    if norms[index] < t0:
        if(index>=(6*set_number) and index<(6*(set_number+1))):
            plt.title('Matched with:'+str(index+1), color='g', fontdict = {'fontsize' : 50})
            plt.imshow(training_tensor[index,:].reshape(height,width), cmap='gray')
            correct_pred += 1
        else:
            plt.title('Matched with:'+str(index+1), color='r', fontdict = {'fontsize' : 50})
            plt.imshow(training_tensor[index,:].reshape(height,width), cmap='gray')
            false_pos +=1
    else:
        if(img_number>=40):
            plt.title('Unknown face!', color='g', fontdict = {'fontsize' : 50})
            correct_pred += 1
        else:
            plt.title('Unknown face!', color='r', fontdict = {'fontsize' : 50})
            false_neg +=1
    plt.subplots_adjust(left = 15, right=20, top=20) #plt.subplots_adjust(right=1.2, top=2.5)
    plt.tick_params(labelleft=False, labelbottom=False, bottom=False,top=False,right=False,left=False, which='both')
    count+=1

fig = plt.figure(figsize=(10, 10))
for i in range(len(testing_tensor)):
    recogniser(i)

plt.show()

print('Correct predictions: {}/{} = {}%'.format(correct_pred, num_images, correct_pred/num_images*100.00))


# In[80]:


print('False positive predictions: ', false_pos)
print('False negative predictions: ', false_neg)
precision= correct_pred/(correct_pred + false_pos)
recall= correct_pred/(correct_pred+ false_neg)
f1= (2*precision*recall)/(precision+ recall)
print('Precision: ', precision)
print('Recall: ', recall)
print('F1 score: ', f1)


# In[81]:


accuracy = np.zeros(len(eigvalues_sort))

def tester(img_number,reduced_data,FP,num_images,correct_pred):
    
    num_images          += 1
    unknown_face_vector = testing_tensor[img_number,:]
    normalised_uface_vector = np.subtract(unknown_face_vector,mean_face)
    
    PEF = np.dot(projected_data,normalised_uface_vector)
    proj_fisher_test_img = np.dot(reduced_data.T,PEF)
    diff  = FP - proj_fisher_test_img
    norms = np.linalg.norm(diff, axis=1)
    index = np.argmin(norms)
    
    set_number = int(img_number/4)

    t0 = 7000000
    
    if norms[index] < t0:
        if(index>=(6*set_number) and index<(6*(set_number+1))):
            correct_pred += 1
    else:
        if(img_number>=40):
            correct_pred += 1
    
    return num_images,correct_pred

def calculate(k):
    
#     print("k:",k)
    reduced_data = np.array(eigvectors_sort[:k]).transpose()
    
    FP = np.dot(projected_sig, reduced_data)
    
    num_images=0
    correct_pred=0
    
    for i in range(len(testing_tensor)):
        num_images,correct_pred = tester(i,reduced_data,FP,num_images,correct_pred)
#     print(FP.shape)
    accuracy[k] = correct_pred/num_images*100.00
    
print('Total Number of eigenvectors:',len(eigvalues_sort))
for i in range(1,len(eigvalues_sort)):
    calculate(i)
    
fig, axi = plt.subplots()  
axi.plot(np.arange(len(eigvalues_sort)), accuracy, 'b')  
axi.set_xlabel('Number of eigen values')  
axi.set_ylabel('Accuracy')  
axi.set_title('Accuracy vs. k-value')


# In[63]:


from PIL import Image
import PIL
for i in [161, 162, 163, 164]:
    base_width = 70
    image = Image.open('Archive2/test_s/' + str(i) + '.jpg')
    width_percent = (base_width / float(image.size[0]))
    hsize = int((float(image.size[1]) * float(width_percent)))
    image = image.resize((base_width, hsize), PIL.Image.ANTIALIAS)
    image.save(str(i)+'.jpg')


# In[64]:


import cv2
for i in [161, 162, 163, 164]:
    base_height = 80
    image = Image.open(str(i)+'.jpg')
    hpercent = (base_height / float(image.size[1]))
    wsize = int((float(image.size[0]) * float(hpercent)))
    image = image.resize((70, base_height), PIL.Image.ANTIALIAS)
    img1 = np.array(image)
    #print(img1.shape)
    image.save(str(i)+'.jpg')
    img = cv2.imread(str(i)+'.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(i)+'.jpg', gray)
    print(gray.shape)
    #image.save(str(i)+'.jpg')


# In[65]:


img = cv2.imread('Archive2/test_s/164.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('Archive2/test_s/164.jpg', gray)


# In[ ]:




