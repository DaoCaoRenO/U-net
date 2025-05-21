def calculate_iou(pred, target, num_classes):
    intersection = (pred & target).float().sum((1, 2))  # Intersection
    union = (pred | target).float().sum((1, 2))  # Union
    iou = intersection / (union + 1e-6)  # Avoid division by zero
    return iou.mean().item()  # Return mean IoU

def visualize_predictions(images, predictions, targets, num_images=5):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].permute(1, 2, 0).cpu().numpy())
        plt.title("Image")
        plt.axis('off')

        plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(predictions[i].cpu().numpy())
        plt.title("Prediction")
        plt.axis('off')

        plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(targets[i].cpu().numpy())
        plt.title("Target")
        plt.axis('off')

    plt.tight_layout()
    plt.show()