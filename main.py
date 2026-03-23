


from model import MNISTNet
from dataset import load_mnist, get_client_data
from training import train, evaluate
from torch.utils.data import DataLoader

trainset, testset = load_mnist()
testloader = DataLoader(testset, batch_size=32)

#Simulate one client training locally
client_loader = get_client_data(trainset, client_id=0)
model = MNISTNet()
model = train(model, client_loader, epochs=3)

accuracy = evaluate(model, testloader)
print(f"Client 0 accuracy: {accuracy:.4f}")
