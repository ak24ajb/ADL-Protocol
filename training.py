
import torch.optim as optim  #optimization algo for weights updation during training




#model: The MNISTNet we built
#trainloader: DataLoader that feeds batches of images and labels
#epochs=1: how many full passes through the dataset

def train(model, trainloader, epochs=1):
    
    #Calculating the Loss Function
    criterion = nn.CrossEntropyLoss()

    #Algo for SGD weight adjustment, lr=Learning Rate
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    #putting the algo in training mode
    model.train()

    #how many times the model's gonna see the images again and again.
    for epoch in range(epochs):
        
        #loading images in batch with their labels
        for images, labels in trainloader:

            #Resets the weights from previous batch
            optimizer.zero_grad()

            #triggers the forward() function from model.py. Returns 10 raw scores per image.
            outputs = model (images)

            # LOSS FUNCTION
            #Compares the model's predictions (outputs) against the correct answers (labels)
            loss = criterion(outputs, labels)

            #Back Propagation to trace back every miscalculation in weights
            loss.backward()

            #Use the gradients to nudge every weight in the direction that reduces the loss
            #Actual learning happens here
            optimizer.step()

    return model


#EVALUTAION FUNCTION
#Testing the model against unseen data for accuracy

def evaluate(model, testloader):
    
    #switch from model.train() to evaluation mode
    model.eval()

    #How many predictions were correct from the total number
    correct, total = 0, 0

    #turn off gradient computation while evaluation
    with torch.no_grad():
        
        for images, labels in testloader:
            outputs = model(images)

            #armax founds out the max score across 10 output neurons
            #If neuron 7 has the highest value, the model is predicting the digit "7".
            
            predicted = outputs.argmax(dim=1)   #dim=1 evaluates this for every image in the batch

            #counting the correct/total predictions
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            
    return correct / total



    
