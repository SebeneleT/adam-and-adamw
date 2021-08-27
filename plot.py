
import matplotlib.pyplot as plt


# loss plot
plt.plot(train_loss_adam, color='green', label='Adam')
plt.plot(train_loss_sgd, color='blue', label='SGDNesterov')
plt.plot(train_loss_adamW, color='red', label='AdamW')
plt.xlabel('Epochs')
plt.ylabel('Train Loss')
plt.legend()
plt.show()


plt.plot(val_loss_adam, color='purple', label='Adam')
plt.plot(val_loss_sgd, color='orange', label='SGDNesterov')
plt.plot(val_loss_adamW, color='cyan', label='AdamW')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend()
plt.show()