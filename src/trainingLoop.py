import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

def trainModel(network, dataset, evalData, epoch):


    #train_data, eval_data = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.75), len(dataset) - int(len(dataset) * 0.75)])

    train_losses, eval_losses = training_loop(network, dataset, evalData, num_epochs=epoch)
    
    for epoch, (tl, el) in enumerate(zip(train_losses, eval_losses)):
        print(f"Epoch: {epoch} --- Train loss: {tl:7.10f} --- Eval loss: {el:7.10f}")

def training_step(network, optimizer, data, targets, kn, lossFn, index):
    optimizer.zero_grad()
    output = network(data)
    #loss = F.mse_loss(torch.masked_select(output[:], data[:, 1] == False, ),
     #              torch.masked_select(targets[:], data[:, 1] == False, ))
    loss = F.mse_loss(output, targets)
    if index % 50 == 0:
        plot(data[0], output[0], kn[0], targets[0], f"loss: {loss}, {index}")
    loss.backward()
    optimizer.step()
    return loss.item()

def eval_step(network, data, targets, known_array, lossFn):
    with torch.no_grad():
        output = network(data)
        #loss = F.mse_loss(torch.masked_select(output[:], data[:, 1] == False,), torch.masked_select(targets[:], data[:, 1] == False,))
        loss = F.mse_loss(output, targets)
        #plot(data[0], output[0], known_array[0], targets[0], loss)
        return loss.item()

def plot(data, output, known_array,target , loss):
    fig, axes = plt.subplots(ncols=4)

    axes[0].imshow(known_array.squeeze().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("known_array")
    axes[1].imshow(data[0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("pixelated Image")
    axes[2].imshow(output.cpu().detach().numpy().squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("prediction")
    axes[3].imshow(target.cpu().detach().numpy().squeeze(), cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("sorce")
    fig.suptitle(loss)
    fig.tight_layout()
    plt.show()
def training_loop(
        network: torch.nn.Module,
        train_data: torch.utils.data.Dataset,
        eval_data: torch.utils.data.Dataset,
        num_epochs: int,
        show_progress: bool = True
) -> tuple[list, list]:
    batch_size = 32
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.001)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=True)
    train_losses = []
    eval_losses = []
    loss_fnc = torch.nn.MSELoss()
    count = 0
    for index in tqdm(range(num_epochs), desc="Epoch", position=0, leave=True):
        network.train()
        epoch_train_losses = []
        for data, known_array, targets in tqdm(train_loader, desc="Minibatch", position=1, leave=False):
            loss = training_step(network, optimizer, torch.concat([data, known_array], dim=1), targets, known_array, loss_fnc, count)

            count += 1

            epoch_train_losses.append(loss)
        train_losses.append(torch.mean(torch.tensor(epoch_train_losses)))

        network.eval()
        epoch_eval_losses = []
        for data, known_array, targets in tqdm(eval_loader, desc="Eval Minibatch", position=1, leave=True, disable=not show_progress):
            loss = eval_step(network, torch.concat([data, known_array], dim=1), targets, known_array, loss_fnc)
            epoch_eval_losses.append(loss)
        eval_losses.append(torch.mean(torch.tensor(epoch_eval_losses)))

    return train_losses, eval_losses