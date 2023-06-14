import os
import json
import random
import numpy as np
import pandas as pd

scriptpath = os.path.abspath(__file__)
APP_ROOT = os.path.dirname(scriptpath)

class AI():
    def __init__(self) -> None:
        # load the CSVs
        # These are prepared by our scripts which can be run through a cron job daily
        self.top_group_clicks = self.load('top_group_clicks.csv')
        self.top_item_clicks = self.load('top_item_clicks.csv')
        self.articles_metadata = self.load('articles_metadata.csv')
        self.user_interactions = self.load('user_interactions_with_groups.csv')
        self.item_clicks = self.load('item_clicks.csv')
        self.group_clicks = self.load('group_clicks.csv')
        # self.clicked_groups = self.group_clicks.loc(columns='group')
        self.images_list = self.load('100k.txt', header=False)
        self.item_embeddings = pd.DataFrame(
            pd.read_pickle(
                os.path.join(APP_ROOT, 'data/articles_embeddings.pickle')
            )
        )

        # load the summary
        with open(os.path.join(APP_ROOT, 'data/summary.json')) as fp:
            self.data = json.load(fp)

        # save individual items
        self.n_items = self.data['n_items']
        self.n_groups = self.data['n_groups']
        self.n_users = self.data['n_users']
        self.n_features = self.data['n_features']
        self.n_clicked_items = self.data['n_clicked_items']
        self.n_clicked_groups = self.data['n_clicked_groups']

    # -----------------------------------------------------------------------
    # API request functions

    def test(self):
        return "Its a test ..."
        
    
    def load(self, filename, header=True):
        """ Loads the csv file
        """
        file_path = os.path.join(APP_ROOT, 'data', filename)
        if header:
            return pd.read_csv(file_path)
        else:
            return pd.read_csv(file_path, header=None)

    def summary(self, key = None):
        """ Returns the summary or the value of any key
        """
        if key is None:
            return self.data

        elif key in self.data.keys():
            return self.data[key]
        
        return None

    def item_image(self, item_id):
        """ returns the image of an item
        """
        if not isinstance(item_id, int):
            item_id = int(item_id)
        
        image = self.images_list.iloc[item_id%100000].values.tolist()[0]
        return f"/static/img/{image}"

    def group_images(self, group_id):
        """ Returns the 5 images of a group (from its recent items)
        """
        items = self.recent_items(group_id=group_id, with_details=False)
        return [self.item_image(item) for item in items]
    
    def group_id(self, item_id) -> int:
        """ Returns the group of an item
        """
        return int(
            self.articles_metadata[
                self.articles_metadata['article_id'] == item_id
            ]['category_id'].values[0]
        )
    
    def clicks(self, item_id = None, group_id = None):
        """ Returns numbre of clicks on an item or a group
        """
        if item_id is not None:
            df = self.item_clicks
            clicks = df[df['item'] == item_id]['clicks'].tolist()
        else:
            df = self.group_clicks
            clicks = df[df['group'] == group_id]['clicks'].tolist()

        return int(clicks[0]) if clicks else 0
    
    def items_count(self, group_id):
        """ Retuns number of items in a group
        """
        return len(
            self.articles_metadata[
                self.articles_metadata['category_id'] == group_id
            ]
        )
    
    def item_details(self, item_id):
        """ Return item details of an item
        """
        if not isinstance(item_id, int):
            item_id = int(item_id)
        
        ic = self.clicks(item_id=item_id)

        return {
            "id" : item_id,
            "image": self.item_image(item_id),
            "category_id": self.group_id(item_id=item_id),
            "time": 1 + (item_id % 19) % 10,
            "comments": ic//19,
            "shares": ic//31,
            "likes": ic*2//3,
            "views": ic,
            "rating": random.randint(1, 5),
        }

    def group_details(self, group_id):
        """ Returns details of a group
        """
        return {
            "id" : group_id,
            "images": self.group_images(group_id),
            "likes": self.clicks(group_id=group_id),
            "article_count": self.items_count(group_id),
            "rating": 5,
        }
    
    def details(self, item_id = None, group_id = None, user_id = None):
        """ Returns the details of an item, group or a user
        """
        if item_id is not None:
            return self.item_details(item_id)
        
        elif group_id is not None:
            return self.group_details(group_id=group_id)
        
        elif user_id is not None:
            return []
        
        else:
            return 'error : you must provide one of item_id, group_id or user_id ...'

    def popular_items(self, group_id = None):
        if group_id is not None:
            group_id = int(group_id)

            # popular items of a group ...
            items = self.user_interactions[
                    self.user_interactions['group'] == group_id
                ].groupby('item').count().sort_values(
                    by='user',
                    ascending=False
                ).index[:5].tolist()
        else:
            items = self.item_clicks[:5].loc[:, 'item'].tolist()

        return [self.item_details(item) for item in items]

    def random_items(self, group_id = None):
        if group_id is not None:
            # random articles of a particular group
            # todo -- 
            return [1, 2, 3, 4, 5]
        else:
            # random articles in general
            n_items = self.summary('n_items')
            items = [random.randint(0, n_items-1) for i in range(5)]
        
        return [self.item_details(item) for item in items]

    def _kNN(self, item_ids:list|int, k=5):
        """ Returns k Near Neighbours to an item
        """
        if not isinstance(item_ids, list):
            item_ids = [item_ids]

        item_ids = [int(item) if not isinstance(item, int) else item for item in item_ids]
        items_vector = np.array(
            self.item_embeddings[
                self.item_embeddings.index.isin(item_ids)
            ].mean()
        )
        distances = np.linalg.norm(self.item_embeddings - items_vector, axis=1)
        kNN = distances.argsort()[1:k+1]
        return kNN
    
    def recommended_items(self, item_id = None, user_id = None):
        if item_id is not None:
            # recommended items based upon an item ...
            items = self._kNN(item_id, k=5)
            return [self.item_details(item) for item in items]
        elif user_id is not None:
            # recommended items for a user ...
            items = self.recent_items(user_id=user_id, with_details=False)
            recent_items = self._kNN(items, k=11)
            items = [item for item in recent_items if item not in items][:5]
            return [self.item_details(item) for item in items]
        else:
            return "error : you must specify an item_id or a user_id ..."

    def recent_items(self, group_id = None, user_id = None, with_details=True):
        if group_id is not None:
            # recent items of a group ...
            group_id = int(group_id)
            items = self.articles_metadata[
                    self.articles_metadata['category_id'] == group_id
                ].sort_values(
                    by='created_at_ts',
                    ascending=False
                ).iloc[:5, 0].tolist()
            
        elif user_id is not None:
            # recent items of a user ...
            user_id = int(user_id)
            items = self.user_interactions[
                    self.user_interactions['user'] == user_id
                ].sort_values(
                    by='timestamp',
                    ascending=False
                ).iloc[:5,1].tolist()
        else:
            # recent items in general ...
            items = self.user_interactions.sort_values(
                    by='timestamp',
                    ascending=False
                ).iloc[:5,1].tolist()
            
        if not with_details:
            return items

        return [self.item_details(item) for item in items]

    def popular_groups(self):
        """ Returns the popular groups
        """
        groups = self.top_group_clicks['group'].values.tolist()
        return [self.group_details(gid) for gid in groups]

    def random_groups(self):
        """ Returns the random groups
        """
        clicked_groups = self.group_clicks['group'].values.tolist()
        l = len(clicked_groups)
        groups = [clicked_groups[random.randint(0, l-1)] for i in range(5)]
        return [self.group_details(gid) for gid in groups]

