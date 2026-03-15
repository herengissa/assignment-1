# Assignment 1: Train, Deploy, Share

**Covers:** L1-L2
**Grading:** G (pass/fail)
**Deadline: Look at the discord channel for assignment 1**
**Submit via:** GitHub Classroom by making commits, and posting the link to your deployment + a statement in a **thread** in assignment_1 channel on discord

## What the heck are we going to do?

Train an image classifier (a neural network) on a dataset we haven't used in class, deploy it as a working API, and share it with the class on Discord so others can try it out.

This is the same process you walked through in L2 - understand, prepare, train, evaluate, iterate, deploy - but on your own, with a different dataset. BUT!!!! All of this will teach you a lot - but there is one particular thing that is important. This assignment is about getting used to doing the homeworks. Moving forward homeworks will be optional projects that you take on to learn that always just give you a starting point. Because I PROMISE YOU, just reading the lesson notebooks and running them DOES NOT MAKE YOU LEARN. It gives you a great idea of concepts, but learning kicks in when someone shows us a white canvas and says “draw”. And then you realize - ok, where do I start?
That is when learning happens. And this lab is to force you into this position.

That means this very first graded assignment is one of many “homeworks” in this course. And the point of the homeworks is to give you a notebook with some basic guidance, which you are going to use to step by step prompt your way to a working model.

## The assignment summarized:

### 1. Train a model

Pick a dataset that wasn't used in the lesson notebooks (not Oxford Pets). The exercise notebook (`lessons/intro/homework/02_exercise_train_your_own.ipynb`) has several options ranging from easy to challenging. You can also find your own dataset.

Your notebook should show:

- **Data exploration** - what does the data look like? How many classes? Is it balanced?
- **Training** - train at least one model using transfer learning
- **Evaluation** - confusion matrix or most confused classes, top losses, sample predictions
- **At least one experiment** - change a hyperparameter (architecture, epochs, image size, etc.), compare the result to your baseline, and explain what happened

### 2. Deploy it

Get your trained model running as an API that accepts an image and returns a prediction. Use the `classifier_deploy/` starter from L2 as a base, or build your own if you prefer.

It should:

- Accept an image upload
- Return the predicted class and confidence
- Run locally with `docker-compose up` (or equivalent)

You don't need to deploy to AWS or any cloud service. Running locally is enough.

### 3. Share it on Discord

Post in the assignment Discord thread:

- A screenshot or short recording of your app making a prediction
- What dataset you used and how many classes
- One thing that surprised you (a weird top-loss image, an unexpected confusion, a hyperparameter that helped or didn't)

Try out at least one classmate's deployment and reply to their post with what happened when you tested it.

## The notebook

You will use the homework notebook from L2: `02_exercise_train_your_own.ipynb`. This notebook is included in the assignment repo. It has the dataset options, the process steps, and empty code cells for you to fill in. That's the whole point - this assignment IS the homework, just graded.

Open the notebook, read through it, pick a dataset, and start prompting your way through each phase. The notebook walks you through the same six steps from L2 (understand, prepare, train, evaluate, iterate, ship) but this time the code cells are blank. Your job is to fill them in, step by step, using an agent to help you write the code.

Don't create a separate notebook from scratch. Work inside this one. The structure is already there - follow it, but remember: different datasets might have different challenges, and the agent will help you navigate that. 

## How to work

Use Claude Code, Cursor, or any AI coding tool throughout. This is encouraged, not just allowed - it's how we work in this course.

But: **don't just accept what the agent gives you.** The goal is to understand what you're building. When the agent writes code, ask it to explain what each part does. When something doesn't work, try to understand why before asking for a fix. When it suggests an approach, ask yourself if it makes sense. 

Good habits from L1:

- **Explain things back to the agent.** "Let me try to explain what data augmentation does - correct me if I'm wrong..." This is one of the best ways to check your understanding.
- **Be critical of outputs.** If the agent generates a DataBlock, read it and make sure you understand each parameter. If something looks off, question it.
- **Ask it to research.** "What resize strategy should I use for satellite images and why?" is better than "write me a DataBlock."
- **Verify.** Always run `show_batch()` before training. Always look at your top losses after training. The agent can't see your outputs - you can.
- You can feed the agent the context of the course plan and the notebooks that will come. After all, this assignment is forcing you to create a machine learning model using a neural network - but we don’t even know how they work, which is fine! Focus on the process, and make sure the agent understands where you are in your journey of learning ML. You are EXTREMLY early!

## What to submit

Your GitHub Classroom repo should contain:

- **The completed homework notebook** (`02_exercise_train_your_own.ipynb`) with all code cells filled in, showing the full process: exploration, training, evaluation, experiment, results
- **A working deployment** (the `classifier_deploy/` folder with your model, or your own setup) with a README explaining how to run it
- **Your exported model** (`.pkl` file) - either in the repo or instructions for where to get it if it's too large for git

## Godkänd

This is pass/fail. To pass, you must do ALL of the following:

- You trained a model on a dataset not used in class
- Your notebook shows you explored the data, trained, evaluated, and ran at least one experiment
- You have a working deployment that accepts images and returns predictions - deployment can be done on any linux server of your choice, as long as its available online. At this point, you should have a small server which you use personally for small projects.
- You posted in the Discord thread,
- If you are aiming for higher grades in this course I expect you to try out the others assigments - why not???? it’s fun to see what cool model they built!!!!
- Your work shows understanding, not just copy-paste - you can explain what your code does if asked. However, at this stage, it’s more about DOING, than understanding exactly how everything works. It’s about the process, not the details. Explore, preprocess, train, evaluate, experiment, etc!
- Try out someone elses model, post your results with the image/s you uploaded.

I will look at your notebooks personally, and I will apply my graderbot agent which will look at various criterias. 

## Tips

- **Start with Imagenette** if you want the easiest path. The DataBlock setup is nearly identical to the lesson. But choose whatever you think sounds fun - the larger the dataset, the better GPU you’ll need, and the more time it will take to train - keep this in mind. CHOOSE ANY DATASET YOUR AGENT CONFIRMS WOULD WORK!
- **Follow the notebook structure.** The homework notebook has phases (understand, prepare, train, evaluate, iterate, ship) with task descriptions and empty code cells. Work through them in order. Each phase builds on the previous one.
- **Start the deployment early.** The ML part is fun but the deployment is where students usually run out of time. Get `docker-compose up` working with the lesson's pet classifier first, then swap in your own model.
- **The experiment section matters.** Don't just train one model and call it done. Change something, compare, and write a sentence about what you learned. This is the habit that makes you an ML practitioner.
- **The Discord post is required.** It's not just for fun - seeing other people's datasets, mistakes, and surprises is genuinely educational.