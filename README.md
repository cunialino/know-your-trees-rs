# Know Your Trees

I would like to implement decision trees algorithms in rust. First thing first, to learn rust.
Secondly, to go through decision trees algrithms in more depth as I use them a lot at work.

## Action Plan

Ideally I would like to implement everything, from id3 to boosting ensembles, but let's define the priorities:

- [ ] CART trees: this is just for learning rust
    - [x] basic cart with numerical features. Gini + Logit Scores
    - [x] Better error handling: I would like to implement error handling instead of unwrapping. 
    Primary reason is unwrap is not really idiomatic, then I would like to learn proper error handling in rust
    - [x] Testing and fixing for null values in features. Should be pretty easy, but you never know
    - [ ] Extend score support
    - [ ] Optimization: we are calculating grad and hess from scratch everytime for target, we could just compute it once.
    - [x] Optimization: parallelize everything on cpu (either tokio or rayon)
    - [ ] Feat: insert categorical features into algo. This is to improve understanding of trees
    - [ ] Feat: implement symmetric trees building. A lot to learn on trees.

- [ ] Ensembles: this should be almost trivial once tree is well defined
    - [ ] Random Forest
    - [ ] Boosting Gradient

- [ ] Distributed computing: this is just for fun
    - [ ] Gpu with thrust cuda + cudarc
    - [ ] Creating a library for pyspark (is this even possible?), would love to implement map + reduce on this.

- [ ] Other trees algorithms: understand tree algos better
    - [ ] ID
    - [ ] c45
