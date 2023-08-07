# Co-HETMRS

> This is the repo for ICRA project

## Checking List

- [ ] Environment

    - [x] Environment Initialization
      - [ ] add store and read part
    - [x] Environment Update and reset
    - [x] Agents' observation
    - [ ] Other heterogeneity setting

- [ ] Preprocessing
  - [ ] Agent
  - [ ] Preprocessor

- [ ] Message Passing and Fusioning



- [ ] RL Framework

# Experiment



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
        C
        D
        C-->D
    end
    
    
    subgraph Individual
         
        B
        A
        A-->B
        E
    end

    direction LR
    B-->C
    D-->E

    F -.-> C
    G -.-> B

    



```