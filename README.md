# Co-HETMRS

> This is the repo for ICRA project

## Checking List

- [x] Environment

    - [x] Environment Initialization
    - [x] Environment Update and reset
    - [x] Agents' observation
    - [ ] Other heterogeneity setting

- [ ] Preprocessing

- [ ] Message Passing and Fusion

- [ ] RL Framework


## Pipeline

```mermaid
graph LR
    
    A[Get Observation]
    B[Preprocessing]
    C[Graph Building]
    D[Aggregation]
    E[Action Selection]
    F[Communication Constraint]
    G[Mobility Constraint]


    subgraph Environment
    F
    G
    end

    subgraph neighborhood
    direction LR
        C
        D
        C-->D
    end
    
    
    subgraph Individual
         
        B
        A
    direction LR
        A-->B
        E
    end

    direction LR
    B-->C
    D-->E

    F -.-> C
    G -.-> B

    



```