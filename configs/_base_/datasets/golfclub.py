dataset_info = dict(
    dataset_name='golfclub',
    paper_info=dict(
        author='Zeng Jianping',
        container='',
        year='2024',
        homepage='',
    ),
    keypoint_info={
        0:
        dict(name='golf_club_head', id=0, color=[255, 0, 0], type='', swap=''),
        1:
        dict(name='golf_club_tail', id=1, color=[0, 0, 255], type='', swap=''),
    },
    skeleton_info={
        0:
        dict(link=('golf_club_head', 'golf_club_tail'), id=0, color=[255, 0, 255]),
    },
    joint_weights=[1., 1.],
    sigmas=[
        0.079,
        0.079,
    ])
