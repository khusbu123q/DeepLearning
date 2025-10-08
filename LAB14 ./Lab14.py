#Vanilla RNN
import numpy as np

def RNN_from_scratch(x1,x2,h_prev,Wxh,Whh,Why):
    inputs=x1,x2
    ht={}
    yt={}
    ht[-1]=np.copy(h_prev)

    for t in range(len(inputs)):

        h_t = np.tanh(np.dot(Whh, ht[t-1] )+ np.dot(Wxh, inputs[t]))
        y_t = np.dot(Why, h_t)

        # Store hidden and output states
        ht[t] = h_t
        yt[t] = y_t

    return ht, yt

def main():
    input_size=2
    hidden_state=3
    output_size=2
    wxh=np.array([[0.5,-0.3],[0.8,0.2],[0.1,0.4]])
    whh=np.array([[0.1,0.4,0.0],[-0.2,0.3,0.2],[0.05,-0.1,0.2]])
    why=np.array([[1.0,-1.0,0.5],[0.5,0.5,-0.5]])
    x1=np.array([[1],[2]])
    x2=np.array([[-1],[1]])
    h_prev = np.zeros((hidden_state, 1))

    print("Running the forward pass of the RNN...")
    hidden_states, outputs = RNN_from_scratch(x1,x2, h_prev, wxh, whh, why)


    print("\nFinal Hidden State at time stamp1:",hidden_states[0])
    print("\nOutput at timestamp1:",outputs[0])

    print("\nFinal Hidden State at time stamp2:", hidden_states[1])
    print("\nOutput at timestamp2:", outputs[1])


if __name__=="__main__":
    main()