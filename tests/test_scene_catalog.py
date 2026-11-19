"""Informational 3D catalogs do not require runtime installation."""


def test_scene_catalogs_are_informational(temp_studio_env):
    client, _ = temp_studio_env
    objects = client.get('/api/3d/catalog').json()['models']
    worlds = client.get('/api/worlds/catalog').json()['models']
    assert {'sf3d','triposr','hunyuan2','hunyuanmini','hunyuan21','trellis2','idol'} == {m['id'] for m in objects}
    assert {'sharp','worldgen','mirror','mirror2','world1','world2','worldsplat'} == {m['id'] for m in worlds}
    assert all('sin integración' in m['state'] for m in objects)
    assert 'UE' in next(m['license'] for m in objects if m['id']=='hunyuan2')
    assert 'research' in next(m['details'] for m in worlds if m['id']=='sharp').lower() or 'investigación' in next(m['details'] for m in worlds if m['id']=='sharp').lower()
