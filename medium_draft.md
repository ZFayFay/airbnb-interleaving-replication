# Interleaving is 50x faster than A/B testing. Sometimes.

## I rebuilt Airbnb's ranking experiments on MovieLens. Their headline number failed my test, then explained itself.

*Part 3 of Big Tech measurement, rebuilt.*

Search teams have a traffic problem. A new ranking model might change bookings by half a percent, and an A/B test needs weeks of traffic to see that, because most searches end without a booking no matter which model is running. At Airbnb it's worse than most places, since people only travel a couple of times a year.

Interleaving is the standard workaround. Instead of splitting users into two groups, you show every user one list that mixes results from both rankers, and you see whose results get booked. Airbnb's search team described their version in [a KDD paper](https://arxiv.org/abs/2508.00751). They pair the rankers' results one slot at a time, flip one coin per search to decide which ranker's item goes first in each pair, and credit a booking to the ranker that contributed the item. They report that it needs about 50 times less traffic than an A/B test, that it agreed with A/B in direction 82% of the time across 29 experiments, and that it can mislead when a ranker optimizes the list as a whole, for example for diversity.

That 50x is the kind of number that ends up in planning documents, so I wanted to see where it comes from.

## How I set it up

Before running anything, I wrote down five hypotheses with pass or fail rules and committed them to git. The world is part real, part simulated. Tastes come from real MovieLens ratings, using the 2,000 most-rated movies and a matrix factorization of what people liked. Each search shows 10 of 50 candidate movies, users pay less attention further down the list, and about 3% of searches end in a "booking". Each user searches twice. I built 12 rankers of very different quality, from a perfect one to random, and compared every pair both ways. Because the booking model is known, I can compute the true A/B effect of every pair exactly, and the number of users each method needs for 80% power.

## The credit scheme is fair

The first check was the paper's claim that the credit scheme has no built-in bias. I simulated users who book purely by position and ignore what the item is, and ran 2,000 interleaving tests on them. 4.3% came out significant at the 5% level, which is what a fair test should do.

## Then the headline number failed

My pre-registered rule was that interleaving should need at least 10 times fewer users than an A/B test for the median pair. It didn't come close. Across all 66 pairs, the median saving was 3.0 times, and the best was 10.5 times. On one pair, two matrix factorization models of almost equal quality, interleaving was ten times *worse* than A/B, because the two lists were very different but neither was better, so every comparison was noise.

Interleaving was right about direction, though. It agreed with the true A/B effect in all 66 pairs, with a correlation of 0.99. So the method worked, but the speedup didn't match the paper.

## Where the 50x comes from

The pattern in the failures was that the speedup was largest when the two rankers were good and similar. That made me think the pairs I'd built were the wrong test. Nobody runs an experiment between a random ranker and a perfect one. Real ranking work compares a model with a slightly improved version of itself.

So I added a test that wasn't in my plan, and I'm labeling it that way. I took one model and made treatments that blend in a small amount of the true appeal, so the two lists share most of their top results. When the lists shared 97% of their top 10, a change worth 1.7% more bookings, interleaving needed 203 times fewer users. At 94% it was 82 times, at 88% it was 32 times, and at 80% it was 15 times.

![Speedup vs overlap](https://raw.githubusercontent.com/ZFayFay/airbnb-interleaving-replication/main/figures/fig1_speedup_vs_overlap.png)

The reason makes sense once you see it. An A/B test pays for all the noise in whether people book at all, which has nothing to do with the change. Interleaving only learns from the slots where the rankers disagree, and the closer the rankers are, the more of the noise it skips. So "50x" isn't a property of interleaving. It's a property of how small Airbnb's changes usually are. It's the same lesson as the geo tests I used to run on the agency side: how sensitive a test is depends on the design and on what's being compared.

## Breaking it

My last pre-registered hypothesis was that I could reproduce the failure the paper describes, where a ranker that optimizes the list as a whole confuses interleaving. I gave each search a hidden intent, a genre the user had in mind, and built a ranker that spreads genres across the top 10. My prediction was that interleaving and A/B would disagree. They didn't. In my world, diversity lost bookings in both, and the more diversity, the bigger the loss. The reason, I think, is that my users judged each movie on its own, so there was no list-level effect for interleaving to miss.

So I added a second test that wasn't in my plan. These users also judge the whole list: the more the top 10 is dominated by one genre, the more likely they leave without booking. Now the methods disagreed every time. In an A/B test, the diverse ranker raised bookings by 7% to 35% depending on how hard it pushed diversity. Interleaving said it lost, at every setting. The merged list is already mixed, so the benefit of diversity disappears, and what's left is a slot-by-slot comparison that the relevance ranker wins.

![List-level effect](https://raw.githubusercontent.com/ZFayFay/airbnb-interleaving-replication/main/figures/fig3_list_level.png)

## What I take from it

I would use interleaving the way Airbnb does, as a fast filter before an A/B test. But I wouldn't plan traffic with a speedup number from someone else's system. The saving depends on how similar the two rankers are, and you can estimate it in advance by measuring how much their lists overlap. And for any change that's about the list as a whole, like diversity, freshness or a new layout, I would skip interleaving and go straight to the A/B test, because that's the case where it can confidently pick the wrong winner.

---

*Notebook and code: [github.com/ZFayFay/airbnb-interleaving-replication](https://github.com/ZFayFay/airbnb-interleaving-replication). The pre-registration is the first commit. The method belongs to Qing Zhang, Alex Deng, Michelle Du, Huiji Gao, Liwei He and Sanjeev Katariya. This is an independent rebuild of their [KDD 2025 paper](https://arxiv.org/abs/2508.00751).*
