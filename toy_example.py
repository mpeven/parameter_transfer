'''
Ideas:
    Try autoencoder for one of the simulated nines
'''



import signal
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torchvision
from datasets import *
from models import ToyModel as Model
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")

# Handle ctrl+c gracefully
signal.signal(signal.SIGINT, lambda signum, frame: exit(0))




def angle_error(y_true, y_pred, num_bins):
    half = num_bins//2
    return half - abs(half - abs(np.array(y_true) - np.array(y_pred)))





def stage_1():
    '''
    Stage 1

    Pre-train CNN to get rotation from simulated 9s
    Train: On simulated 9s at random rotations
    Test:  On MNIST 9s at random rotations (to make sure it's working)
    '''
    num_bins = 360
    model = Model(num_bins).to(DEVICE)
    loss_func = torch.nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)



    phase_dataloaders = [
        ("train", torch.utils.data.DataLoader(Nines("mnist", "train"), batch_size=256, num_workers=1)),
        ("test", torch.utils.data.DataLoader(Nines("mnist", "test"), batch_size=256, num_workers=8))
    ]
    num_epochs = 200
    print_every = 1
    best_test_acc = 0
    best_avg_error = 361
    for epoch in range(1, num_epochs):
        if epoch % print_every == 0:
            print("Epoch {:03d}/{:03d}".format(epoch, num_epochs))
        for phase, dl in phase_dataloaders:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_losses = []
            running_thetas = []
            running_preds = []
            for sim in tqdm(dl, desc="{}".format(phase), ncols=115, leave=False):
                images = sim['image'].to(DEVICE)
                thetas = sim['theta'].to(DEVICE)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass + Loss calculation
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                loss = loss_func(outputs, thetas)

                # Backward pass + Optimization
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_losses.append(loss.item())
                running_thetas.extend(sim['theta'].numpy())
                running_preds.extend(preds.cpu().numpy())

            if epoch % print_every == 0:
                print("{} \t Loss={:05.2f}, Accuracy={:05.2f}, Avg Error={:05.2f}".format(
                    phase,
                    np.mean(running_losses),
                    100*np.mean(np.equal(running_thetas, running_preds)),
                    np.mean(angle_error(running_thetas, running_preds, num_bins)),
                ))

            if phase == "test":
                test_acc = 100*np.mean(np.equal(running_thetas, running_preds))
                avg_error = np.mean(angle_error(running_thetas, running_preds, num_bins))
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                if avg_error < best_avg_error:
                    best_avg_error = avg_error
                    best_model = model

    print("Best accuracy = {:05.2f}".format(best_test_acc))
    print("Best avg error = {:05.2f}".format(best_avg_error))
    return best_model





def stage_2(model):
    '''
    Stage 2

    Minimize MSE between real and simulated with respect to the simulated parameters
    Note: Need to get a monte-carlo estimation of the gradient
    '''
    angle_to_sim = 115

    print("Trying to get the simulator to start producing 9's at {} degrees".format(angle_to_sim))
    real_dl = torch.utils.data.DataLoader(Nines("mnist", "test", angle_to_sim), batch_size=1, num_workers=8)
    sim_dl = torch.utils.data.DataLoader(Nines("simulated", "test"), batch_size=360, num_workers=8, shuffle=False)

    loss_func = torch.nn.CrossEntropyLoss().to(DEVICE)
    model.eval()
    running_dist = []

    # Estimate a distribution of the loss for each real image
    i = 0
    for real in tqdm(real_dl, ncols=115):
        outputs = model(real['image'].to(DEVICE))
        _, theta_real = torch.max(outputs, 1)
        theta_real = theta_real.cpu().numpy()[0]
        # print("Theta real = {}".format(theta_real))
        # print("\t (actual = {})".format(real['theta'].numpy()[0]))

        # Find the distribution: theta_real - theta_sim
        for sim in sim_dl:
            images = sim['image'].to(DEVICE)
            thetas = sim['theta'].to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
            dist_array = angle_error(np.repeat(theta_real, 360), preds, 360)

            running_dist.append(dist_array)

        i += 1
        if i == 200:
            break

    all_err = np.concatenate(running_dist)
    x = np.mod(np.arange(all_err.shape[0]), 360)
    df = pd.DataFrame({'angle': x, 'loss': all_err})
    print(df)
    pd.set_option("display.max_rows",360)
    print(df.groupby('angle').mean())
    sns.lineplot(x="angle", y="loss", err_style="band", ci=99.9, estimator="mean", data=df)
    plt.show()





def main():
    model_weight_path = "trained_model.pt"

    trained_cnn = stage_1()
    torch.save(trained_cnn.state_dict(), model_weight_path)

    trained_cnn = Model(360).to(DEVICE)
    trained_cnn.load_state_dict(torch.load(model_weight_path))
    stage_2(trained_cnn)




if __name__ == '__main__':
    main()
