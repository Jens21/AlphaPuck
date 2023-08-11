import laserhockey.hockey_env as h_env
import numpy as np
import ray

from own_env import OwnEnv
from action_selection import ActionSelection
from network import Network

# evaluates the performance of the current network version against strong and weak opponents, for 100 episodes each
# is called by the trainer actor
class Evaluator():
    def __init__(self, gamma, shared_memory, tree_search_depth):
        # fixed the seeds to make results comparable
        self.seeds = [
            # 12328514, 49829056, 22838017, 38491996, 82268821, 59494661,
            # 82908488, 24549929, 96438603, 5090968, 46853013, 79381653,
            # 97055922, 25096521, 21508805, 24394666, 7096212, 32000391,
            # 89222738, 29029228, 34997161, 87994109, 42605842, 26283810,
            # 34166218, 19105544, 52326605, 10263566, 52411238, 44111082,
            # 14956654, 22728504, 96481732, 29111778, 39359376, 85209258,
            # 62048583, 21191525, 19970799, 28437311, 61580869, 2460329,
            # 12736675, 4737109, 80652251, 73703193, 2408915, 83537208,
            # 34553951, 75849449, 19772937, 61536417, 94776543, 56472659,
            # 94455311, 15088179, 25985926, 20177874, 63420328, 71498254 # till here 60 seeds

            25404460, 67476337, 99167134, 96679725, 48841914, 33245639,
            6811971,  6491264, 10704239, 21174847, 86758116, 14159511,
           47332795, 67623915, 17931643, 70229679, 10143948, 63220668,
           76996122, 55050161, 87150011, 23908637, 40601426, 17034750,
           11005085,   551421, 99012765, 34274438, 15928048, 43979604,
           28671773, 11927587, 22145464, 24172980, 16305631, 33662500,
           50825644, 93920600,  2449397, 33883637, 48605549,  4888641,
            4587877, 91790736, 84331817, 23607684, 16562666, 41577001,
           59795090, 53702195, 38147733,  9087800, 95609191, 58616471,
           44862873, 78788210, 78185757, 25234564, 84347909, 69101559,
           56515057, 17855007, 22113104, 83008098, 54946453, 93845731,
           32181134, 83000204, 33363212, 39939105, 16266802, 47759313,
            9800640, 99515055, 57820736, 31533230, 67500536, 80203784,
           21348276, 95219777, 60644509, 47592339, 64936309, 79964285,
           30111158, 49015911, 28406940, 43731061, 14861206, 19112822,
           15160746, 48061429, 17043726,  1339702, 41493073, 22892188,
           89271129, 95090028, 76695473, 48754913, 18658986, 20308835,
           72268341, 81198751, 68674347, 31642659, 30940396, 12446462,
           64318924, 87628437, 14073424, 86048139, 49169402, 26800392,
           94236558, 36118811, 49759935,   934458, 98329410, 16792407,
           54299691, 43200380,  1744148, 14366046, 40581913, 63988673,
           75363304, 95086302, 49491816, 38170253, 25438536, 23630716,
           50948440, 38151139, 21543075, 99034225, 56332368, 25393975,
            9474811, 35665189,  4289561, 81849183, 10293326, 29158517,
           93172454, 62509711, 97085676, 16161183, 48252411,  6993006,
           92233346, 37967052, 53622178, 63615275, 91248709, 60834151,
           53247286, 73834559,  2972929, 86623578, 57176506, 60890353,
           77859259, 59996002, 43413785, 49283554, 58330507, 58159053,
           64226988, 51820105, 13020051, 33591095,  4311760, 89327124,
           84687498, 32565228, 43518637, 93538086, 65766147, 59395400,
           98116692, 26521535, 35742178, 71252437, 51696143, 24354807,
           52629532, 93604571, 94060806, 20854729, 67551432, 59156757,
           20561964, 38817312, 26920752, 98971251, 40160764, 76680795,
           99507284, 81806257  # till here 200 seeds
        ]

        self.gamma = gamma
        self.shared_memory = shared_memory
        self.tree_search_depth = tree_search_depth

        self.action_selection = ActionSelection(self.gamma)

    def evaluate(self, render=False):
        # gets the current model from the shared memory
        net = Network()
        net.load_state_dict(ray.get(self.shared_memory.get_current_model.remote()))

        env = OwnEnv()

        l_winners_weak = []
        l_winners_strong = []
        for i, seed in enumerate(self.seeds):
            env.seed(seed=seed)
            obs_1, _ = env.reset(customized=False) # didn't use customization, since not needed

            use_weak = (True if i>len(self.seeds)//2 else False)
            opponent = h_env.BasicOpponent(weak=use_weak)

            while True:
                action_1, _, _ = self.action_selection.select_action(0., net, obs_1, True)

                # altough we train with frameskip >= 1, evaluation happens with no frameskip (similar in tournament)
                for _ in range(1):
                    if render:
                        env.render()
                    action_2 = opponent.act(env.obs_agent_two())
                    obs_1, reward, done, trunc, info = env.step(np.hstack([action_1, action_2]), customized=False)

                    if trunc:
                        break

                if trunc:
                    if use_weak:
                        l_winners_weak.append(info['winner'])
                    else:
                        l_winners_strong.append(info['winner'])
                    break

        env.close()

        l_winners_weak, l_winners_strong = np.array(l_winners_weak), np.array(l_winners_strong)

        n_won_weak, n_won_strong = (l_winners_weak == 1).sum(), (l_winners_strong == 1).sum()
        n_draw_weak, n_draw_strong = (l_winners_weak == 0).sum(), (l_winners_strong == 0).sum()
        n_lost_weak, n_lost_strong = (l_winners_weak == -1).sum(), (l_winners_strong == -1).sum()

        return n_won_weak, n_won_strong, n_draw_weak, n_draw_strong, n_lost_weak, n_lost_strong, len(l_winners_weak), len(l_winners_strong), net