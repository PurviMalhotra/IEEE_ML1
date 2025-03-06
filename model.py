import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

def load_dataset_from_csv(csv_path, image_size=(28, 28), test_size=0.2, random_state=42):
        data=pd.read_csv('/Users/purvimalhotra/Developer/ccode/IEEE_ml1/data.csv')   #load the provided dataset
        print(f"Loaded dataset with shape: {data.shape}")                            #shape of the dataset
        
        y=data.iloc[:, 0].values                                        #labels
        X=data.iloc[:, 1:].values                                       #pixel values
        
        X=X.reshape(X.shape[0],image_size[0],image_size[1])             #reshape into images
        
        X_train,X_test,y_train,y_test=train_test_split(                 #splitting into train and test sets
            X,y,test_size=test_size,random_state=random_state,stratify=y
        )
        
        return (X_train, y_train),(X_test, y_test)
    
    
csv_path="/Users/purvimalhotra/Developer/ccode/IEEE_ml1/data.csv"  

(X_train,y_train),(X_test,y_test)=load_dataset_from_csv(csv_path)

print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Testing data shape:", X_test.shape)
print("Testing labels shape:", y_test.shape)

class_names=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

plt.figure(figsize=(10,10))
for i in range(min(25,len(X_train))):                              #displaying sample images
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]] if y_train[i] < len(class_names) else f"Class {y_train[i]}")
plt.suptitle('Sample Images from Dataset',fontsize=16)
plt.show()

sample_image=X_train[0]                                          #verifying grayscale format
print("Shape of a single image:",sample_image.shape)
print("Min pixel value:", np.min(sample_image))
print("Max pixel value:",np.max(sample_image))

plt.figure(figsize=(6,6))
plt.imshow(sample_image, cmap='gray')
plt.colorbar()
plt.grid(False)
plt.title('Grayscale Image Example')
plt.show()

#EDA
                                                               #displaying one example for each class
unique_classes=np.unique(y_train)
num_classes=len(unique_classes)
print(f"Number of unique classes: {num_classes}")

cols=min(5, num_classes)                                        #adjusting grid acc to number of classes
rows=(num_classes + cols-1) // cols
plt.figure(figsize=(cols*3,rows*3))

for i,class_idx in enumerate(unique_classes):
    idx=np.where(y_train==class_idx)[0][0]  
    plt.subplot(rows,cols,i+1)
    plt.imshow(X_train[idx], cmap='gray')
    class_name=class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"
    plt.title(f'Class {class_idx}: {class_name}')
    plt.axis('off')
plt.suptitle('One Sample from Each Class', fontsize=16)
plt.tight_layout()
plt.show()

                                                              #class distribuition 
class_counts=np.bincount(y_train)
plt.figure(figsize=(12, 6))
sns.barplot(x=list(range(len(class_counts))),y=class_counts)
plt.xticks(range(len(class_counts)), 
           [f"{i}: {class_names[i]}" if i < len(class_names) else f"Class {i}" 
            for i in range(len(class_counts))], 
           rotation=45)
plt.title('Class Distribution in Training Set')
plt.xlabel('Class')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

pixel_stats=pd.DataFrame({                                  #stats for pixel values
    'min': np.min(X_train, axis=(0, 1)).flatten(),
    'max': np.max(X_train, axis=(0, 1)).flatten(),
    'mean': np.mean(X_train, axis=(0, 1)).flatten(),
    'std': np.std(X_train, axis=(0, 1)).flatten()
})
print("Pixel value statistics:")
print(pixel_stats.describe())

                                                            #preprocess data
X_train_norm=X_train / 255.0
X_test_norm=X_test / 255.0

#reshape for ml models
X_train_reshaped=X_train_norm.reshape(X_train_norm.shape[0], -1)      
X_test_reshaped=X_test_norm.reshape(X_test_norm.shape[0], -1)

#logistic regression model
print("Training Logistic Regression model...")
logreg=LogisticRegression(max_iter=1000,solver='saga',n_jobs=-1)
logreg.fit(X_train_reshaped, y_train)

#predictions and evaluations 
y_pred=logreg.predict(X_test_reshaped)
logreg_accuracy=accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {logreg_accuracy:.4f}")

#classification report
unique_labels=np.unique(np.concatenate([y_test, y_pred]))
target_names=[class_names[i] if i < len(class_names) else f"Class {i}" for i in unique_labels]
print("\nClassification Report for Logistic Regression:")
print(classification_report(y_test,y_pred,target_names=target_names))

#confusion matrix visualization for logistic regression
plt.figure(figsize=(12, 10))
cm=confusion_matrix(y_test, y_pred)
tick_labels=[class_names[i] if i < len(class_names) else f"Class {i}" for i in unique_labels]
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=tick_labels,yticklabels=tick_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Logistic Regression')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

device=torch.device("cpu")

#Neural Network        
class FashionClassifier(nn.Module):
    def __init__(self, input_dim,hidden_dim1,hidden_dim2,output_dim):
        super(FashionClassifier, self).__init__()
        self.flatten=nn.Flatten()
        self.fc1=nn.Linear(input_dim,hidden_dim1)
        self.relu1=nn.ReLU()                                        #ReLU action
        self.dropout1=nn.Dropout(0.2)
        self.fc2=nn.Linear(hidden_dim1,hidden_dim2)
        self.relu2=nn.ReLU()          
        self.dropout2=nn.Dropout(0.1)                               #dropout for regularization        
        self.fc3=nn.Linear(hidden_dim2,output_dim)
        
    def forward(self,x):
        x=self.flatten(x)
        x=self.dropout1(self.relu1(self.fc1(x)))
        x=self.dropout2(self.relu2(self.fc2(x)))
        x=self.fc3(x)
        return x

n_classes=max(np.max(y_train),np.max(y_test)) + 1                    #number of classes for output layer
print(f"Number of classes for neural network: {n_classes}")

X_train_tensor=torch.FloatTensor(X_train_norm).unsqueeze(1)          #prepare data for pytorch and pytorch tensors
X_test_tensor=torch.FloatTensor(X_test_norm).unsqueeze(1)
y_train_tensor=torch.LongTensor(y_train)
y_test_tensor=torch.LongTensor(y_test)

#creating dataset and dataloaders
train_dataset=TensorDataset(X_train_tensor, y_train_tensor)
test_dataset=TensorDataset(X_test_tensor, y_test_tensor)

batch_size=128
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

#initialize model
input_dim=X_train_norm.shape[1] * X_train_norm.shape[2]            #flatenning input dimensions
model=FashionClassifier(input_dim=input_dim,hidden_dim1=128,hidden_dim2=64,output_dim=n_classes)
model=model.to(device)

print("\nNeural Network Summary:")
print(model)

criterion=nn.CrossEntropyLoss()                                    #loss function and optimiser
optimizer=optim.Adam(model.parameters(),lr=0.001)

#training function
def train_model(model,train_loader,criterion,optimizer,num_epochs=10):
    model.train()
    train_losses=[]                                          #initialization
    train_accs=[]
    
    for epoch in range(num_epochs):
        running_loss=0.0                                     #loss for epochs
        correct=0                                            #correct predictions
        total=0                                              #total sample
        
        for inputs,labels in train_loader:
            inputs,labels=inputs.to(device),labels.to(device)
            
            optimizer.zero_grad()                            #Zero the parameter gradients
                                                             #Prevents gradients from accumulating across batches   
            outputs=model(inputs)                            #Forward pass: compute model predictions
            loss=criterion(outputs, labels)                  #Compute the loss between predictions and true labels
            
            loss.backward()                                  #Backward pass: compute gradients
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)     #update running loss
            _, predicted=torch.max(outputs.data, 1)
          ;  total += labels.size(0)                          #updating total samples and corect predictions
            correct += (predicted == labels).sum().item()
        
        epoch_loss=running_loss / total
        epoch_acc=correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
    
    return train_losses, train_accs

#evaluation function
def evaluate_model(model, test_loader, criterion):
    model.eval()                                           #Set the model to evaluation mode
                                                           #This disables layers like dropout and batch normalization
    #initialize variables again
    test_loss=0.0
    correct=0
    total=0
    all_preds=[]
    all_labels=[]
    
    with torch.no_grad():
        for inputs,labels in test_loader:                  #iterate though test data batches
            inputs,labels=inputs.to(device),labels.to(device)
            
            outputs=model(inputs)                         #Forward pass: get model predictions
            loss=criterion(outputs, labels)
            
            test_loss += loss.item() * inputs.size(0)
            _, predicted=torch.max(outputs.data, 1)       #Get the predicted class (highest probability)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss=test_loss/total
    test_acc=correct/total
    
    return test_loss, test_acc, all_preds, all_labels

#training
print("\nTraining Neural Network:")
train_losses,train_accs=train_model(model,train_loader,criterion,optimizer,num_epochs=10)

#evaluating
test_loss,test_acc,y_pred_classes,y_test_np=evaluate_model(model,test_loader,criterion)
print(f"\nNeural Network Test Accuracy: {test_acc:.4f}")
print(f"Neural Network Test Loss: {test_loss:.4f}")

#plotting history
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_accs,label='Training Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_losses,label='Training Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

#confusion matrix for neural network
plt.figure(figsize=(12,10))
cm_nn=confusion_matrix(y_test_np,y_pred_classes)
sns.heatmap(cm_nn,annot=True,fmt='d',cmap='Blues',xticklabels=tick_labels,yticklabels=tick_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Neural Network')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#classifiaction report for neural network
print("\nClassification Report for Neural Network:")
print(classification_report(y_test_np,y_pred_classes,target_names=target_names))

print("\nModel Comparison:")
print(f"Logistic Regression Accuracy: {logreg_accuracy:.4f}")
print(f"Neural Network Accuracy: {test_acc:.4f}")

num_samples=min(15, len(X_test))
plt.figure(figsize=(12,8))
for i in range(num_samples):
    plt.subplot(3,5,i+1)
    plt.imshow(X_test[i],cmap='gray')
    true_class=y_test[i]
    pred_class=y_pred_classes[i]
    true_label=class_names[true_class] if true_class < len(class_names) else f"Class {true_class}"
    pred_label=class_names[pred_class] if pred_class < len(class_names) else f"Class {pred_class}"
    color='green' if true_class==pred_class else 'red'
    plt.title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=8)
    plt.axis('off')
plt.suptitle('Neural Network Predictions', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

torch.save(model.state_dict(),'fashion_classifier_pytorch.pth')
print("Model saved as 'fashion_classifier_pytorch.pth'")

def load_model_and_predict(model_path,input_image):

    loaded_model=FashionClassifier(input_dim=input_dim,hidden_dim1=128,hidden_dim2=64,output_dim=n_classes)  #load model
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()
    
    if len(input_image.shape)==2:                                      #preprocess image
        input_image=input_image.reshape(1,1,input_image.shape[0],input_image.shape[1])
    elif len(input_image.shape)==3:
        input_image=input_image.reshape(1,input_image.shape[0],input_image.shape[1],input_image.shape[2])
    
    input_tensor=torch.FloatTensor(input_image)                        #convert to tensor
    
    with torch.no_grad():                                              #get predictions
        output=loaded_model(input_tensor)
        _, predicted=torch.max(output, 1)
    
    return predicted.item()

print("\nExample of using the saved model for prediction:")
sample_idx=42                                                         #sample image for test set
sample_image=X_test_norm[sample_idx]
sample_label=y_test[sample_idx]
predicted_class=load_model_and_predict('fashion_classifier_pytorch.pth',sample_image)

plt.figure(figsize=(6,6))
plt.imshow(X_test[sample_idx],cmap='gray')
true_name=class_names[sample_label] if sample_label < len(class_names) else f"Class {sample_label}"
pred_name=class_names[predicted_class] if predicted_class < len(class_names) else f"Class {predicted_class}"
result="Correct" if sample_label==predicted_class else "Incorrect" 
plt.title(f"True: {true_name}\nPredicted: {pred_name}\nResult: {result}")
plt.show()
