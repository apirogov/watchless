# Learn Git Rebase in 6 minutes // explained with live animations!
- URL: https://www.youtube.com/watch?v=f1wnYdLEpgI
- Channel: The Modern Coder
- Track via: api, en-US (uploader)

## Summary

**TL;DR:** Git rebase replays feature commits on top of updated master to create a linear history, and the video demonstrates a team workflow for syncing, rebasing, fast-forwarding master, and pushing—while warning that rebase rewrites commit IDs and is risky on shared branches.

- [overview] The video is a short animated Git tutorial with terminal commands and graph animations illustrating merge versus rebase.
- [claim] Merge combines feature commits into a merge commit on master, which can make team history messy; rebase instead replays feature commits on top of master for a straight-line graph. (no source)
- [claim] Rebase rewrites history by creating new copies of commits, so commit IDs change even when author date and contents stay the same. (no source)
- [claim] Rebase is less suitable for open-source pull requests and risky on shared branches because it rewrites commits; the video points to a linked Atlassian caveats article. (named: linked Atlassian article)
- [procedure] Workflow: git pull on master; git checkout -b my_cool_feature; commit; git checkout master; git pull; git checkout my_cool_feature; git rebase master; resolve conflicts; git checkout master; git rebase my_cool_feature; git push.

**Assessment:** The speaker promotes his own LearnGit.io course and grounds the workflow mainly in his team’s Amazon experience; the caveats are delegated to a linked Atlassian article, and the demo uses “master” rather than newer “main” naming.

## Description

|
🧠 Build a full mental model of Git (it's free btw) → https://app.learngit.io/home/?v=rebase

I built LearnGit.io as a more complete curriculum. In the AI era, these mental models are going to matter more than command memorization. No pressure, just wanted to get the word out. Full access is also free for students!

---

DESCRIPTION
In this git rebase tutorial, I'll show you a different way of approaching git branching/merging with rebase. My team at Amazon adopted the workflow you'll see in the video and we love it. Whether you're mostly a solo developer or on a large team, give this a shot, I guarantee that even if you don't use rebase all the time, it will become an essential part of your development toolkit.

CONCEPTS
In this video:
 • Rebase vs. merge: what's the difference?
 • Full rebase workflow for team collaboration
 • When to use rebase (and when not to)
 • Commands: git rebase, git pull, git push

Workflow shown:
 • Sync with remote master
 • Create feature branch
 • Rebase feature onto updated master
 • Rebase master onto feature (fast-forward)
 • Push to production

LINKS
Written rebase reference: https://www.themoderncoder.com/a-better-git-workflow-with-rebase/
Rebase caveats: https://www.atlassian.com/git/tutorials/merging-vs-rebasing#the-golden-rule-of-rebasing

MY TECH
https://www.themoderncoder.com/wfh-nyc-desk-tour-2022/

MUSIC
Intro/outro music by Dyalla: https://soundcloud.com/dyallas

## Transcript

Hey what is going on guys it is Jack from the Modern Coder and on this episode of Coding Tip of the Day I'm rocking my classic Maryland Ultimate Jersey.

Shout out to my Ultimate Frisbee people.

I even got some of my favorite discs on the wall over there.

Anyway on this episode I'm going to be talking about a better Git workflow with Rebase instead of Merge.

So I'll start off by explaining the basic concepts of what's a Rebase, what's a Merge, and then I'll run through the actual Git commands and the steps for how a Rebase-based workflow might look.

So right out of the gate here the goal of both merging and rebasing is to take commits from a feature branch and put them onto a master branch or any other branch for that matter.

So let's start with how a quote-unquote normal merge would happen.

So let's say I have a graph that looks like this. As you can see I split off from my feature branch at commit two of master and I've done a bit of work on my feature branch with those two commits there.

Now when I want to merge those two changes back onto my master branch I'm going to run a Git merge and what this will do it will take all of my feature branch changes, all the changes from both of those commits, and kind of stuff them into one merge commit and then put that merge commit on master or main line.

So after those changes are combined together into that merge commit your graph might look something like this and if you're working on team with other developers you can imagine that this could kind of get gross looking kind of maybe like this where you have a bunch of people working on different feature branches and they've been pushing changes and stuff like that.

So this is a little bit gross to look at it's confusing to trace and so this is where rebase is going to help us out.

So let's jump in and see how rebase would work. So let's rewind a little bit.

So instead of doing a Git merge I'll do a rebase when I want to take my two feature branch commits and move them onto master.

What a rebase does is it takes all of your commits from your feature branch and basically moves them on top of the master commits like this.

So behind the scenes what Git is doing is actually blowing away the feature branch commits and duplicating them as new commits on top of the master branch.

So in a sense it's rewriting history and if I open up my Git log here before my rebase and then after my rebase you can see that though the commits author date and the contents of the commit itself are the same the commit IDs are different meaning that it's created a new copy of that commit when we ran our rebase.

And what you get with this approach is a nice clean straight line graph with all of your commits laid out nicely in a row.

It makes it really easy to trace what commits went where and as you can imagine if you're on a team with a lot of developers all of the commits are still in a row so it's really easy to follow even if you have a bunch of people working on the same project at the same time.

At this point I think I better mention some drawbacks to rebase because nothing is 100% perfect.

Rebase doesn't play super well with open source projects and pull requests just because it can be hard to trace the especially small changes introduced to a code base and this point's a little bit nuanced but I left a link to an article in the description that I think does a good job explaining it.

And secondly it can also be a little bit dangerous if you're working on a shared branch with other developers just because rebasing like I talked about earlier actually rewrites commits but in the workflow example I'm about to go over I'm going to show you a way to basically mitigate this risk and for the development team I've worked on we've successfully adopted the rebase workflow I'm about to show you and we love it.

So that's the basics of rebase so let's now review what this might look like when you actually put it into practice.

So when you first start development you want to make sure you always sync the remote master to make sure you have all the changes that your teammates have been has been doing on your local machine so you just want to run a "git pull" and make sure you're up to date.

Great so I'm already up to date so when I start developing a feature you always want to check out a new branch so that all your commits can all your commits and code can live on this branch without disrupting your master so we're just going to "git checkout -b" and we'll say you know "my_cool_feature". Great!

So now I'm on a feature branch now I'm on my feature branch.

So as I'm developing along I'm going to just I'm going to be periodically committing okay so at this point I've developed and committed two commits to my feature branch.

So as I'm making commits on my feature on my local machine, I also have my fellow developers who are shipping features to our remote mainline or our master branch.

So this means that my development environment is kind of out of sync with the remote mainline and that's okay.

We'll fix that when we want to merge.

Great so say now that I'm now I'm done developing my feature and I want to merge it back into master so what I want to do is check out my master "git checkout master" (my local master) and I want to do a "git pull" so what this will do is it will make sure that my local master is up to date with the remote master and any of the changes that my co-workers may have pushed in the meantime. As you can see I'm one commit behind.

So now that I have all of my co-workers changes on my local master, I want to make sure that my feature branch will jive with those changes.

And so what I want to do here is "git checkout my_feature_branch" And I'll do a "git rebase master" against the master branch.

What this will do is it will re-anchor my feature branch against the latest changes.

And at this point git will let me know if I have any conflicts and if I do I can take care of them and commit all my code to my feature branch.

In this case I have no conflicts so I'm good to go.

So now that my feature branch doesn't have any conflicts with my up-to-date master, I'm going to go ahead and check it out, "git checkout master", and rebase master now against my feature branch "git rebase my_cool_feature".

And what this will do is it will take those commits on my feature branch and replay them on top of my master branch so now my local main line (my local master branch) is nice and straight and ready to be pushed to production.

So at this point I should be good to go and all I have to do is a "git push".

So if you never used the rebase workflow before I know a lot of those specific commands might be a little confusing but don't worry, check out the link in the description I've written up a blog post with everything we talked about this video and so head over there if you want to take like have a reference for when you actually go and try out this rebase workflow.

So I think that'll do it for today's episode, as always if you have a question leave it in the comments and if you have a suggestion for the next coding tip the day leave it in the comments as well.

Leave a like if you found this video helpful and yeah that is it I'll see you the next one.

Peace!
