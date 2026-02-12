This is where I am writing what I want in raw language :

Prompt : 

I am giving you this transcipt in which a person is explaining hydra , or say it giving tutorial on hydra to start using it in the repo , for better organsiation and stuff..  ,based on what is taught by him in the transcript or lets say foloowing that way or be around that in hydra , mastering whole hydra thing is not useful as well as essential in my work too , this guy was from my team telling all other how to use hydra in training runs.. , for now its imp for me to fug out to use hydra in my training runs and to learn the stuff he said + additional oe level advanced so that U can do more cool suffs okay plus well knowledged , so I want a whole tutorial from very basics till here (one level up that what the guy was talking in conversation) , but I am a kind of person who only understands when context is well setu up like what was the need for this, why to use , hydra docs are themselves not that user firendly to read and undersatdn I didn't get what why they are doing what easy ness is there and also the language is not super simple.. , like there is not enough text to explain what things they are doing and why , why like this like why dot dot method , why picking up names via files inside folder , the deafult settings and all , all is writeen in code only.. , I was not able to undersatdn it through docs , so i want you to develop a website interactive docs as per my usecase which I have usecase upon , not like advanced super , but around the conversation and a level up and taking into account and knowldege from original hydra documention okay . One point to note is this conversation is transcribed from a speech to text model and person was speaking in hingnlish , so text in transcript may be not perfectaerly captres but basically this was just the base like telling you where i want stuff on hydra. Main is I want you to design interactive personides hydra from where I can read and learn hydra as per how I learn stuff , because I don;t find hydra docs easy to undersatnd , I usually understand anything when there reasons for why , more examples shown to do stuff , purpose , significane and use raw sentences and words instead of big jargons, to learn stuff ofc I would need code and complex things but road to that should be clear in what it is made to underatnd , see you already knew how much and in what way I discuss stuff from you from past chat discussions also you can refer from that in crafting these intercative custom docs. Secondly clearly reference is novasrplus repo , the way implemeted in this I would need to implemnent in taht way in my furher training runs , so what I want to learn is basics plus how in novasrplus it is implemented , so blend of undersatdning + whats there in novasrplus and I would be able to write my won training runs using it..

Conversation recording : 

00:00
Why Hydra click if you actually
00:47
Okay so when the past like we have used Jason fine and then regular exec Python files to do config management. I am not saying like this is the best way to do it but I really like, you know, this is kind of like the sinarest way to go about doing things.
01:17
The one thing that I like about is it is, it's click composable and it opens up a lot of new ways to Lake combines about job for think and also, reduce code, basically. So you don't have to leg through manner, documentation, and Lake changing the same file again and again.
01:45
So Hydra documentation Hai shit. I mean the first time I added even, I didn't understand like why was the need for it and Hydra is just not for deploying. Like it is also used in the one of the places if people generally used in simple. So these are like the standard why Hydra kind of thing.
02:12
Like you have a conflict instead of like writing it writing everything manually. You can override directly from sale. I like you we trained file and you know change any manual, any parameter that you want to change then you can like with this one to you. So then it also auto says that the conflict that you have Run you can change it.
02:45
Wherever you want to like in one be conflict. So I will just show you a basic example. So we just have like two files. One is a conflict file. I know the one is a 10 file. So the train file is a standard lightning example that anybody can look at.
03:11
Like, this is a simple CNN, a tiny as you net, and we have like, you know, a dictionary of models with all the things and then we have a trainer with model like the model for your function training, step, optimizers Hai, and then we have legumin function, which runs at okay.
03:32
So basically how do you like get the conflict in Lake? What is the entry point for the conflict? So this decorator essentially, like gives you the conflict path which is the root of this file. What is the Name of the conflict file. So this could, this could be like, you know, experiment one.
03:55
And this this can be experiment like this. Sorry, experiment one. And you could have like this versions that we don't need that. Then what it essentially does Lake. You have this cfg. This config will essentially get you the moderate lips. Cfg.model type would get you. The models cag.mora.s channels will get you the channels and the rest of the parameters that you want to train and just surrounded, essentially, that is how a simple flat file configuration.
04:33
Is there. So at the rate Hyderabad main, this is the entry point Lake descript. That you run training. Where is it? Coming? Configu name is basically like its automatically picking up e x P1. That's a name whatevery name. So generally what I am I do is generally my experiment name is also my conflict name, which is past tense live automatically so there is a should be automatically.
05:05
Yes. But you can also do something like this. So say suppose. So we have to give mention all those. You can also do it like this. Like if I go src 01 example, dot 5 - C so I can. So if I have another file that I want to write lyrics experiment to theek hai ismein, maine Kuchh changes kiye.
05:39
Hain ki number off, classes is 10, I am doing hundred legs. See for hundred kar raha hun main. Okay, so I will have this and I will just say xp2 so it will automatically pick up your speed to and run it, okay, how does it know from where to pick up?
05:56
So that is the magic that I drop jo hai, na Yahan, per hai, theek hai. So generally like whenever you Rider train file, so you will have a standard conflict file. So, in my case, there is src Hydra hyphen conflict, which is already provided in the conflict path. So, any time I do minus C exv2 in automatically pick up that conflict path and transit for me.
06:22
So, any flight flats five that you have in? Graphic file is ideally like, I am I consider it as my experiment. Okay, so that's the standard, you know, flat file basic wala thing, okay? So yeah. So, Like we don't have any folder.
08:05
Train dot simple CNN. So the file Python file dot, simple CNN. So it automatically pics of the entire thing and passes passes. That object on to your conflict, we have to build that Trial dot simple CNN. So if it is resonant then it is trail.rini, vessel basically train dot need to have a path right now.
08:41
That network is not there in trained, or it will be model dot care. So that's how you I am thinking. Sin, citizen trying file on your already. There is no need of Management. Default nothing is their such in the same file inventor. So, I generally don't use this structure.
09:08
Like, I have introduced structure, have used the dictionary, wala structure mostly I use the destructor, like the model, wala things in my own trainer. I find it Lake simple to deal with other than this target. Wala. I mean everybody has its there on everyone should have the same thing.
09:37
Okay, so that was the thing. That will be problem attic when we have so many networks and also So this was a marimal, conflict file, the next one is Lake, if you have like multiple data sets, okay? So how do you actually go about and do things, okay? So you like this is like, very simple, the data set could be like, you know, something directly downloaded from FIFA Aur ite koi bhi.
10:25
Look at data set. So this train or local data set is nothing but my data loader that I have return in train, not parameters. Okay, so Yah target Hai yah, main Python mein, import karta, hun, Aise ka, hissa, aur FIR usko, yah parameter, deta hun, but instead of doing that, you just can like next wala, step to you can compose I have together.
11:09
Vah kaise, likhna hai vah chahie, not happening to do this. Maine Ek, comment likha tha
11:41
Read this, please write it in English or write a code skin. That help you convert everything. Okay? So it's so you don't have to late manually go about and changes and most of the changes that we make will be safe function changes. Like, you are changing the learning rate and testing it out.
12:15
You are changing your schedule and chair checking it out. So you do once you have everything said by the cloud and you are, you feel happy that okay this is the configuration that I want to use, you can just do it. Like I will give example to it. You will write data set, dot sifaten Yahan per jo, bhi naam, hai Data loader essential, whatever.
12:53
Anybody like data set dot, Han like whatever is your file name and parts. So if you, if you actually called this is the data loader, okay?
13:47
Okay. And also in the conflict if I have a different learning, so you will have to add it like this, it will be something like optimizer. Optimizer and then, you know, you change that. So, Adam optimizer Hai. You can just change these things. Learning rate, Aisa change Karke. You can just run the experiment hai.
14:22
Okay, so addam ka kya tha 001 so it was actually I can do this. And then later on this experiment directly, Okay, you don't need to like copy paste, all the things again and changes. So it loads this and then we patched using justice. So that's how you generally like change.
14:45
If you want to change anything, you just change it like this. Dusra dikhana padega, Ek Hai commands. So as you can see Yahan per just multimode model, tiny resident data, set Hai. So these things like yah jo hai These are this is like you know, your overriding the defaults which you can do it directly here.
15:17
I am overriding LR pay optimizer.lr just translate that first Command ko, pahle main aapko, se batao to comment. Jarur. Okay, so your model is basically a model simple CNN. Sorry. Yeah. So the model is here so it's been map to this. Okay, the optimizer that you have added is being map of your weight.
15:50
Sorry data set jo hai. So data set is here. Sifatan. Okay. And the optimize that we have as Adam is Adam. Cure if I change if I want to change the optimizer I just have to do add and w. So where is the states store? Post running? Because like that, you can ask in this one line command, it will save the so.
16:17
But let's go to compossible, we don't want to use this structure. Also, you can like change it like using this. Okay. Wherever you want to save it. Okay then, okay, so this is basically like a flat file composer compossible. So what we are trying to do is we have a base file where we are mapping.
16:43
This is not complete. Okay, so we have a simple CNN and the channels and they had in and stuff. If you want to do an experiment V1, okay? Say, suppose you want to change justice to parameters, you just you know, default import the base. Okay, change the experiment name and self is essentially like okay.
17:19
Okay selfish basically merge order control. Like if you if you write this like you know, you Lord the base config and then you apply this in basically like you know lets you in this in this year are some conflicts that you want to change it so you add that, okay.
17:41
And then experiment. So then essentially, like if you want to find you just basically name and these extra parameters and then, you know, change it similar with experiment, B2 so back and change the basic self and trying. So, one thing like once you like, make sure that you have a set base line that you can rely on and then you can live, which is why we are also not good to use this thing.
18:16
Easy to the baselines is five, should be easy, not to touch the base file, baseline file so which I hope is the compoble working. Yes. So compossible work flow. We have the same things. We have the data sets. I have added losses. Basically, then you have the models and you have the optimizes, okay?
18:43
So the base Amal is basically life. You know, your loading all the defaults. So it will. If your name is diffused also because I don't know, tested it So basically like, you know you have your defaults you just loaded simple Si and and model message here. Simple, CNN Yahan, the data set.
19:11
You have to add a optimizer and you have a loss called cross entropy, okay? And then, basically, you just add all these things. Abhi yah Jo model simple, CNG vagaira hai na ite ise. Not Window optimizer in the base ka. Folder ite ise. Simple. CNN Hai Uske andar hai na.
19:42
Simple Si Hai Models, Naam based dot yarmm. Liye Hai Uske. Andar Matlab Main Agar model Van likhunga aur Jahan per main morning Hai vahan per model Van use karna, chalu kar. Dunga to vah ran Karenge model, Ham kahan, modern man, Nahin Hai left side, mein Ki Hai, right side, mein value, Ho Jaega.
20:32
Jo Bhi Tum feature off, folders keys are all basically Hydra cratifieds. These are Hydra jo bhi Ki Hai Hydra, ke chij. Hai doston Idea ka jo reference skin, aur use, hota hai. Vah use Karoge light side, mein can be the Python ka things Unse Kuchh, import Karke, La rahe.
20:52
Ho that is what you use. So tourist, Take a chair from somewhere.
21:22
We don't want to use enable this games precisely, the one that we want to use precisely isn't disable. Guru bhai Sar confidence. Confidence. Theek hai every folder computer, defaulter Kiya bus. Editing thing you don't have to do it because it is inside this theek hai overall default Mein Sab before Jaega.
21:53
This is how you structure, theek hai. So you do not end up accidentally editing the base time because you never have to edit the placement. Everything else is separate model mein. I want to add a last model. Default is a steam take steaming data set and disable open data set, locate the default.
22:23
Theek hai specified. So I don't even have to specify some base one. You do not have to exit because if you wanted to make changes to other things, you would have to go to base file. Do not do that more. The default of you don't. Specify everything so on your migrating, the data set data.
22:48
So this is the latest. I just want this, this is change in different from the older one only that only one. Arrested conflict spelling, Bahar aata. Tha abhi Main Usko. Andar le aaya hun, explain it some other time. Why? But look at that, and look at how this conflicts are being used inside the coal, that electro also look Do not just look at the complete folder and what the last statement you said, you have to see how this is being used inside the code, and make sure you understand the full that you are any.
23:36
So, there is kya Kaun Omega conference, your referred to everything a strings, right? Some and you need a mechanism to instantiate them is Python objects. Omega kaun se kam se. That's what it will do for you so that I to decom final winner Kaun, you will have to use
24:04
We can do dot dot dot, right. Instead of packets quotes. Syntax.
24:35
Yah circuit style hai friends. Look at the code is not very complicated. Do bar, padhoge.
24:51
Use this pattern to maybe. Conflict that I row. No, I will talk about that function, register reserves, use that to create experiment names. Okay, and if you using tense about this, will give the one week and experience.
25:35
Thank u.

Links/paths essential in context to prompt, my expectations, and conversation transcript. 
novasrplus repo path : /home/amanag/novasrplus

Hydradocs homepage link do web search or whatever to extract info relevent for making interactive docs : https://hydra.cc/docs/intro/
