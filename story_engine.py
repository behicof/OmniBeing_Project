
class MarketStoryWeaver:
    def __init__(self):
        self.stories = []

    def weave_story(self, market_event):
        if market_event["type"] == "crash":
            story = f"Once the market was filled with euphoria, until fear swept through, wiping gains and hearts alike."
        elif market_event["type"] == "rally":
            story = f"A timid optimism turned into roaring confidence as prices soared beyond resistance and logic."
        elif market_event["type"] == "reversal":
            story = f"The market, once deceiving with false hope, turned back on its path, teaching traders a costly lesson."
        else:
            story = f"The market moved sideways, reflecting the indecision of minds and the calm before the storm."

        self.stories.append(story)
        return story

    def get_all_stories(self):
        return self.stories
