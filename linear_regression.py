from matplotlib.animation import FuncAnimation
import numpy as np

class Linear_Regression:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.m = 0 # this is the slope of the line
        self.b = 0 # this is the y_intercept of the line
        self.learning_rate = 0.001
    
    def fit(self, epochs):
        self.epochs = epochs
        # let's create some empty list to store the values during the training
        self.m_values = []
        self.b_values = []
        self.cost_values = []

        # now we can start the "training"
        for _ in range(epochs):
            y_pred = self.m * self.x + self.b # this is the prediction
            # let's see the error or cost
            cost = (1 / len(self.x)) * sum(val**2 for val in (self.y - y_pred))
            # let's check the derivative with respect to m and b
            dm = -(2 / len(self.x)) * sum(self.x * (self.y - y_pred))
            db = -(2 / len(self.x)) * sum(self.y - y_pred)
            # let's update the values
            self.m = self.m - self.learning_rate * dm
            self.b = self.b - self.learning_rate * db
            # now we can store the values in the list that we have created before
            self.m_values.append(self.m)
            self.b_values.append(self.b)
            self.cost_values.append(cost)

    def plot_algorithm(self):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        
        # create a figure
        fig = plt.figure(figsize = (10, 8))
        # add subplot
        ax = fig.add_subplot(111)
        ax.scatter(self.x, self.y, color = 'red')
        line, = ax.plot(self.x, self.m_values[0] * self.x + self.b_values[0], color = 'blue')
        # add some text to better understand the visual
        cost_text = ax.text(0.02, 0.95, '', transform = ax.transAxes)
        iteration_text = ax.text(0.02, 0.90, '', transform = ax.transAxes)
        slope_text= ax.text(0.02, 0.85, '', transform = ax.transAxes)
        y_intercept_text = ax.text(0.02, 0.80, '', transform = ax.transAxes)
        learning_rate_text = ax.text(0.02, 0.75, '', transform = ax.transAxes)
        epochs_text = ax.text(0.02, 0.70, '', transform = ax.transAxes)
        data_points_text = ax.text(0.02, 0.65, '', transform = ax.transAxes)

        # now we can create a function to update the plot 
        def update_plot(i):
            line.set_ydata(self.m_values[i] * self.x + self.b_values[i])
            cost_text.set_text(f'Cost: {self.cost_values[i]}')
            iteration_text.set_text(f'Iteration n. : {i}')
            slope_text.set_text(f'Slope: {self.m_values[i]}')
            y_intercept_text.set_text(f'Y intercept: {self.b_values[i]}')
            learning_rate_text.set_text(f'Learning Rate: {self.learning_rate}')
            epochs_text.set_text(f'Tot Epochs: {self.epochs}')
            data_points_text.set_text(f'n. Data Points: {len(self.x)}')

            return line, cost_text, iteration_text, slope_text, y_intercept_text, learning_rate_text, epochs_text, data_points_text

        # now it's possible to create the animation
        anim = FuncAnimation(fig, update_plot, frames = len(self.m_values), interval = 100, blit = True)

        # save the plot
        anim.save('line.gif', writer = 'imagemagick', fps = 100)

        # show the plot
        plt.show()

# run the program
if __name__ == '__main__':
    # create a set of xs and ys
    x = np.array([1,2,3,4,5,6,7,8,9,10])
    y = np.array([2,4,6,8,10,12,14,16,18,20])

    # create the model 
    model = Linear_Regression(x, y)

    # train the model
    model.fit(epochs = 100)

    # plot the algorithm
    model.plot_algorithm()

'''It's working!'''


