def evaluate_model(model,num_epochs,loss_vals, l2_loss_vals, validation_loss, l2_lambda, corr_vals, alpha = 0.1, reg_flag = True):    
  '''
  Calls the helper plotting functions to plot various evaluation metrics for the models
  '''
  lossplot(num_epochs, loss_vals, l2_loss_vals,l2_lambda, validation_loss, reg_flag) # Training los
  plot_loglosses(num_epochs, loss_vals, l2_loss_vals,l2_lambda, validation_loss, reg_flag)  # PLot loss on log scale
  plot_inferred_vs_actual_params(model, alpha = alpha)
  plot_output_encoder_vs_decoder(model, alpha = alpha)
  plot_correlation_params(model,alpha = alpha)
  plot_correlation_outputs(model, alpha = alpha)
  plot_corr_vs_epochs(num_epochs,corr_vals)

    

def plot_inferred_vs_actual_params(model,alpha = 0.1):
  '''
  Scatterplot of the orginal params with the inferred params
  '''
  with torch.no_grad():
    original  = inputs[:,-1].cpu()
    inferred = model.h_activity_vec.detach().cpu()
    
  ip_size = original.shape[0]
  x = np.arange(0,ip_size)

  plt.scatter(x, original, marker = 'o', label= 'Ground truth')
  plt.scatter(x, inferred,alpha = alpha, label = 'Inferred')
  plt.xlabel("Parameter")
  plt.ylabel("Value")
  plt.title("Ground truth v/s inferred parameters")
  plt.legend()
  plt.show()



def plot_loglosses(num_epochs, loss_vals, l2_loss_vals,l2_lambda, validation_loss, reg_flag):
    
  plt.plot(np.linspace(1,num_epochs, num_epochs).astype(int), loss_vals, label = 'training loss')

  if reg_flag: 
      plt.plot(np.linspace(1,num_epochs, num_epochs).astype(int), l2_loss_vals, label = 'regularization loss')
    
  plt.plot(np.linspace(1,num_epochs, num_epochs).astype(int), validation_loss, label = 'validation loss')
  plt.title("Training and regularization loss (log scale) (lambda = {0})".format(l2_lambda))
  plt.xlabel("Number of epochs")
  plt.ylabel("Loss")
  plt.yscale('log')
  plt.legend()
  plt.show()


# DEPRECATED: Not in use
def val_lossplot(epochs,val_loss):
    plt.plot(np.linspace(1,epochs, epochs).astype(int), val_loss, label = 'Validation loss')
    plt.title("Validation loss")
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    

def lossplot(epochs, loss_vals, l2_loss_vals,l2_lambda, val_loss, reg_flag ):
  '''
  Plot the tranining and regularization losses versus the number of epochs
  '''
  plt.plot(np.linspace(1,epochs, epochs).astype(int), loss_vals, label = 'Training loss')
  if reg_flag:
      plt.plot(np.linspace(1,epochs, epochs).astype(int), l2_loss_vals, label = 'Regularization loss')
  plt.plot(np.linspace(1,epochs, epochs).astype(int), val_loss, label = 'Validation loss')
  plt.title("Training and regularization loss (lambda = {0})".format(l2_lambda))
  plt.xlabel("Number of epochs")
  plt.ylabel("Loss")
  # plt.xscale('log')
  plt.legend()
  plt.show()


def plot_output_encoder_vs_decoder(model,alpha = 0.1):
  '''
  Scatterplot of the output of the encoder vs the decoder
  '''
  with torch.no_grad():
    encoder_output  = output.cpu()
    decoder_output = model(input_w_time_hidden).detach().cpu()

  ip_size = encoder_output.shape[0]
  x = np.arange(0,ip_size)
    
  # Enforce same size
  plt.scatter(x, encoder_output, label= 'Ground truth output')
  plt.scatter(x, decoder_output,alpha = alpha, label = 'Decoder output')
  plt.xlabel("Output")
  plt.ylabel("Value")
  plt.title("Ground truth v/s decoder output")
  plt.legend()
  plt.show()
    

def plot_correlation_params(model,alpha = 0.1):
  '''
  Plot the correlation between the inferred and ground truth parameter vectors
  '''
  with torch.no_grad():
    original  = inputs[:,-1].cpu()
    inferred = model.h_activity_vec.detach().cpu()
  
  plt.scatter(original, inferred)
  plt.title('Correlation between the inferred and ground-truth parameters')
  plt.xlabel('Ground truth parameters')
  plt.ylabel('Inferred parameters')
  plt.show()

    
    
def plot_correlation_outputs(model, alpha = 0.1):
  '''
  Plot the correlation between the encoder output and the decoder output
  '''
  with torch.no_grad():
    encoder_output  = output.cpu()
    decoder_output = model(input_w_time_hidden).detach().cpu()
  
  plt.scatter(encoder_output, decoder_output)
  plt.title('Correlation between the encoder and decoder outputs')
  plt.xlabel('Ground truth output')
  plt.ylabel('Decoder output')
  plt.show()
    
    
def run_lambda_variation(lambda_list, norm,epochs):
    
    test_loss = []
    train_loss = [] 
    for norm_lambda in lambda_list:
        
        # Instantiate the network 
        model_decoder =neuron_unit_decoder_w_hidden_init_fix().to(device)
        train_loss_vals, _, val_loss_vals = train_decoder_w_hidden(model_decoder, loss_fn, l_norm = norm, n_epochs = epochs ,n_lambda = norm_lambda)
        test_loss.append(val_loss_vals[-1])
        train_loss.append(train_loss_vals[-1])
        print(f'Trained model on lambda: {norm_lambda}\n')
    
    return train_loss, test_loss

def plot_lambda_vs_loss(lambda_list, test_L_lambda, train_L_lambda, norm, epochs):
    
    plt.plot(lambda_list[::-1],test_L_lambda, label = "validation loss")
    plt.plot(lambda_list[::-1],train_L_lambda, label = "training loss")
    plt.xlabel("$\lambda$")
    plt.ylabel("Loss")
    plt.xscale('log')
    plt.title(f'Losses as a function of lambda \n {norm} norm, {epochs} epochs')
    plt.legend()
    plt.show()
    
def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

def plot_corr_vs_epochs(epochs,corr_vals):
    plt.plot(np.linspace(1,epochs, epochs).astype(int), corr_vals, label = '$R^2$ coefficient')
    plt.title("Correlation v/s training time")
    plt.xlabel("Number of epochs")
    plt.ylabel("$R^2$ Coefficient")
    plt.legend()
    plt.show()