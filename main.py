
import copy

from model import MNISTNet
from dataset import load_mnist, get_client_data
from training import train, evaluate
from torch.utils.data import DataLoader

trainset, testset = load_mnist()
testloader = DataLoader(testset, batch_size=32)

#Simulate the servers global model
global_model = MNISTNet()

#Client makes a deep copy
local_model = copy.deepcopy(global_model)
client_loader = get_client_data(trainset, client_id=0)

#Client trains locally
local_model = train(local_model, client_loader, epochs=1)

#Extract the weight update
update = {}
for key in global_model.state_dict():
    
    #calculating the Delta
    update[key] = local_model.state_dict()[key] - global_model.state_dict()[key]

# Print the update shape for one layer to verify
print("Update shape for fc1.weight:", update['fc1.weight'].shape)
print("Mean update magnitude:", update['fc1.weight'].abs().mean().item())


#1st simulation-Simulate one client training locally
#client_loader = get_client_data(trainset, client_id=0)
#model = MNISTNet()
#model = train(model, client_loader, epochs=3)
#accuracy = evaluate(model, testloader)
#print(f"Client 0 accuracy: {accuracy:.4f}")
