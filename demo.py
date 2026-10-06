import math
import random
import threading
import time

from flask import Flask, jsonify, render_template_string

from brain import SurvivalBrain
from world import SurvivalWorld

app = Flask(__name__)
LOCK = threading.Lock()

world = SurvivalWorld()
brain = SurvivalBrain()

running = False
shifted = False
episode = 1
shift_episode = None

last_action = "WAIT"
last_reward = 0.0
last_event = "READY"
episode_reward = 0.0
reward_history = []


def perception(w):
    ax, ay = w.agent
    fx, fy = w.food
    dx, dy = w.danger
    scale = max(1, w.size - 1)

    return [
        (fx - ax) / scale,
        (fy - ay) / scale,
        (dx - ax) / scale,
        (dy - ay) / scale,
        w.energy / 100.0,
    ]


def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def new_position(w, avoid):
    while True:
        pos = [
            random.randint(0, w.size - 1),
            random.randint(0, w.size - 1),
        ]
        if pos not in avoid:
            return pos


def apply_action(w, action):
    if not w.alive:
        return None

    x, y = w.agent

    if action == "UP":
        y -= 1
    elif action == "DOWN":
        y += 1
    elif action == "LEFT":
        x -= 1
    elif action == "RIGHT":
        x += 1

    x = max(0, min(w.size - 1, x))
    y = max(0, min(w.size - 1, y))
    w.agent = [x, y]

    w.energy -= 1
    w.steps += 1
    event = None

    if not shifted:
        if w.agent == w.food:
            w.energy = min(100, w.energy + 30)
            w.score += 10
            event = "ENERGY COLLECTED"
            w.food = new_position(w, [w.agent, w.danger])

        elif w.agent == w.danger:
            w.health -= 35
            w.score -= 10
            event = "DANGER HIT"
            w.danger = new_position(w, [w.agent, w.food])

    else:
        if w.agent == w.food:
            w.health -= 35
            w.score -= 10
            event = "OLD FOOD IS DANGER"
            w.food = new_position(w, [w.agent, w.danger])

        elif w.agent == w.danger:
            w.energy = min(100, w.energy + 30)
            w.score += 10
            event = "NEW ENERGY COLLECTED"
            w.danger = new_position(w, [w.agent, w.food])

    if w.energy <= 0:
        w.health -= 5

    if w.health <= 0:
        w.alive = False

    return event


def calculate_reward(old_agent, old_food, old_danger, w, event):
    reward = -0.02

    old_f = distance(old_agent, old_food)
    new_f = distance(w.agent, old_food)
    old_x = distance(old_agent, old_danger)
    new_x = distance(w.agent, old_danger)

    if not shifted:
        reward += 0.20 if new_f < old_f else (-0.10 if new_f > old_f else 0.0)
        reward += -0.15 if new_x < old_x else (0.05 if new_x > old_x else 0.0)
    else:
        reward += 0.20 if new_x < old_x else (-0.10 if new_x > old_x else 0.0)
        reward += -0.15 if new_f < old_f else (0.05 if new_f > old_f else 0.0)

    if event in ("ENERGY COLLECTED", "NEW ENERGY COLLECTED"):
        reward += 2.0

    if event in ("DANGER HIT", "OLD FOOD IS DANGER"):
        reward -= 2.0

    if w.agent == old_agent:
        reward -= 0.05

    if not w.alive:
        reward -= 2.0

    return reward


def reset_episode():
    global world, episode, episode_reward, last_reward, last_event
    world = SurvivalWorld()
    episode += 1
    episode_reward = 0.0
    last_reward = 0.0
    last_event = "NEW EPISODE"


def simulation_loop():
    global last_action, last_reward, last_event, episode_reward

    reward_for_brain = None

    while True:
        time.sleep(0.16)

        with LOCK:
            if not running:
                continue

            senses = perception(world)
            action = brain.think(
                senses,
                reward=reward_for_brain,
                done=False
            )

            old_agent = world.agent.copy()
            old_food = world.food.copy()
            old_danger = world.danger.copy()

            event = apply_action(world, action)

            reward = calculate_reward(
                old_agent,
                old_food,
                old_danger,
                world,
                event
            )

            reward_for_brain = reward
            last_action = action
            last_reward = reward

            if event:
                last_event = event
            elif reward > 0:
                last_event = "POSITIVE SIGNAL"
            elif reward < -0.1:
                last_event = "NEGATIVE SIGNAL"
            else:
                last_event = "LEARNING"

            episode_reward += reward

            if not world.alive or world.steps >= 300:
                try:
                    brain.think(
                        perception(world),
                        reward=reward,
                        done=True
                    )
                except RuntimeError:
                    pass

                reward_history.append({
                    "episode": episode,
                    "reward": round(episode_reward, 2),
                    "shifted": shifted,
                })

                if len(reward_history) > 40:
                    reward_history.pop(0)

                reset_episode()
                reward_for_brain = None


def averages():
    if shift_episode is None:
        before = [x["reward"] for x in reward_history]
        return {
            "before": round(sum(before) / len(before), 2) if before else None,
            "early": None,
            "recent": None,
            "recovery": None,
        }

    before = [
        x["reward"] for x in reward_history
        if x["episode"] < shift_episode
    ]

    after = [
        x["reward"] for x in reward_history
        if x["episode"] >= shift_episode
    ]

    early = after[:3]
    recent = after[-3:]

    before_avg = sum(before) / len(before) if before else None
    early_avg = sum(early) / len(early) if early else None
    recent_avg = sum(recent) / len(recent) if recent else None

    recovery = None
    if early_avg is not None and recent_avg is not None:
        recovery = recent_avg - early_avg

    return {
        "before": round(before_avg, 2) if before_avg is not None else None,
        "early": round(early_avg, 2) if early_avg is not None else None,
        "recent": round(recent_avg, 2) if recent_avg is not None else None,
        "recovery": round(recovery, 2) if recovery is not None else None,
    }


HTML = r"""
<!DOCTYPE html>
<html>
<head>
<title>CADENCE // SURVIVAL</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(circle at 50% 10%,#111d35 0%,#070b13 38%,#030507 100%);color:#fff;font-family:Consolas,"Courier New",monospace;min-height:100vh}
.topbar{min-height:84px;border-bottom:1px solid #263247;display:flex;align-items:center;justify-content:space-between;padding:0 34px;background:rgba(5,8,14,.96)}
.title{font-size:25px;font-weight:bold;letter-spacing:4px}.title span{color:#5ee7ff}.status{color:#65ff9a;font-size:12px;letter-spacing:2px}
.container{width:96%;max-width:1600px;margin:24px auto;display:grid;grid-template-columns:minmax(520px,1.25fr) minmax(390px,.75fr);gap:18px}
.panel{background:rgba(8,13,22,.95);border:1px solid #27344a;border-radius:10px;overflow:hidden}
.panel-title{min-height:50px;display:flex;align-items:center;justify-content:space-between;padding:0 18px;border-bottom:1px solid #27344a;color:#aab8ce;font-size:12px;letter-spacing:2px}
.arena-area{padding:24px}.legend{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:18px}.legend-box{padding:12px;border:1px solid #1f2d43;border-radius:6px;background:#080d16;text-align:center}
.legend-symbol{font-size:20px;margin-bottom:5px}.legend-label{color:#71839e;font-size:10px;letter-spacing:1px}.legend-role{margin-top:5px;font-size:11px}
.cyan{color:#5ee7ff!important}.green{color:#68ff91!important}.red{color:#ff526b!important}
.arena{width:min(62vh,570px);height:min(62vh,570px);max-width:100%;margin:auto;display:grid;grid-template-columns:repeat(10,1fr);grid-template-rows:repeat(10,1fr);gap:3px;background:#101725;border:1px solid #33445f;padding:3px}
.cell{background:#090e17;border:1px solid #121c2b;display:flex;justify-content:center;align-items:center;font-size:22px}
.agent{color:#5ee7ff;text-shadow:0 0 13px #5ee7ff}.food{color:#68ff91;text-shadow:0 0 13px #68ff91}.danger{color:#ff526b;text-shadow:0 0 13px #ff526b}
.shifted-food{color:#ff526b;text-shadow:0 0 13px #ff526b}.shifted-danger{color:#68ff91;text-shadow:0 0 13px #68ff91}
.right{display:flex;flex-direction:column;gap:18px}.brain-flow{padding:20px}.flow-title{color:#657995;text-align:center;font-size:10px;letter-spacing:2px;margin-bottom:10px}
.perception-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}.sense{background:#080d16;border:1px solid #1c2a3e;border-radius:5px;padding:9px 10px}
.sense-name{color:#70829d;font-size:9px;letter-spacing:1px}.sense-value{margin-top:4px;font-size:15px}.flow-arrow{text-align:center;color:#5ee7ff;font-size:20px;padding:9px 0}
.brain-row{display:flex;align-items:center;justify-content:center;gap:18px}.brain-orb{width:125px;height:125px;flex-shrink:0;border-radius:50%;border:1px solid #5ee7ff;background:radial-gradient(circle,rgba(94,231,255,.28),rgba(94,231,255,.03) 67%);box-shadow:0 0 40px rgba(94,231,255,.23);display:flex;justify-content:center;align-items:center;text-align:center;color:#5ee7ff;letter-spacing:2px;font-size:11px;transition:.2s}
.brain-orb.active{box-shadow:0 0 60px rgba(94,231,255,.55);transform:scale(1.03)}
.decision-box{flex:1;border:1px solid #29405a;border-radius:6px;background:#09111d;padding:15px;text-align:center}.decision-small{color:#71839e;font-size:9px;letter-spacing:2px}.decision-action{color:#5ee7ff;font-size:23px;margin-top:8px;font-weight:bold}
.metrics{padding:0 20px 20px}.metric{display:flex;justify-content:space-between;padding:9px 0;border-bottom:1px solid #182235;color:#8fa0b8;font-size:12px}.metric strong{color:white}
.controls{padding:15px}button{width:100%;padding:13px;margin:5px 0;background:#111a29;color:white;border:1px solid #33445f;border-radius:6px;font-family:inherit;letter-spacing:1px;cursor:pointer}button:hover{border-color:#5ee7ff;color:#5ee7ff}.shift{border-color:#ff526b;color:#ff7388}
.bottom{grid-column:1/-1;padding:20px}.proof-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:18px 0}.proof-card{border:1px solid #1f2d43;border-radius:6px;background:#080d16;padding:14px}.proof-name{color:#71839e;font-size:9px;letter-spacing:1px}.proof-value{font-size:20px;margin-top:7px}
.graph{height:205px;display:flex;align-items:center;gap:5px;padding:25px 4px 25px;position:relative}.zero-line{position:absolute;left:0;right:0;top:50%;height:1px;background:#33445f}.bar-wrap{flex:1;height:100%;position:relative;min-width:18px}.bar{position:absolute;left:2px;right:2px;min-height:3px;background:#5ee7ff;box-shadow:0 0 7px rgba(94,231,255,.3)}.bar.positive{bottom:50%}.bar.negative{top:50%;background:#ff526b}
.ep-label{position:absolute;bottom:-20px;left:0;right:0;text-align:center;color:#61728b;font-size:9px}.shift-marker{position:absolute;top:4px;bottom:18px;width:2px;background:#ff526b;box-shadow:0 0 10px rgba(255,82,107,.6);z-index:5}.shift-marker span{position:absolute;top:-18px;left:7px;white-space:nowrap;color:#ff7186;font-size:9px;letter-spacing:1px}
.regions{display:flex;justify-content:space-between;color:#657995;font-size:10px;letter-spacing:2px;margin-top:4px}.footer-note{margin-top:14px;color:#60718a;font-size:11px}
.shift-alert{position:fixed;inset:0;z-index:100;background:rgba(8,0,5,.94);display:none;justify-content:center;align-items:center;text-align:center}.shift-alert.show{display:flex}.shift-box{color:#ff526b;border:1px solid #ff526b;padding:48px 65px;background:#09050a;box-shadow:0 0 90px rgba(255,82,107,.28)}.shift-box h1{letter-spacing:5px}.shift-box p{color:white;line-height:1.8}
@media(max-width:950px){.container{grid-template-columns:1fr}.bottom{grid-column:1}.proof-grid{grid-template-columns:1fr 1fr}}
</style>
</head>
<body>

<div class="shift-alert" id="shiftAlert">
  <div class="shift-box">
    <h1>⚠ WORLD SHIFT DETECTED</h1>
    <p>TARGET MEANINGS HAVE CHANGED</p>
    <p>● F &nbsp; SEEK → AVOID<br>▲ X &nbsp; AVOID → SEEK</p>
    <p>SAME CADENCE BRAIN<br>NO MODEL RESET</p>
  </div>
</div>

<div class="topbar">
  <div class="title">CADENCE <span>//</span> SURVIVAL</div>
  <div class="status">● CADENCE BRAIN ONLINE</div>
</div>

<div class="container">

<div class="panel">
  <div class="panel-title"><span>SURVIVAL ARENA</span><span id="worldMode">WORLD // NORMAL</span></div>
  <div class="arena-area">
    <div class="legend">
      <div class="legend-box"><div class="legend-symbol cyan">◆</div><div class="legend-label">AGENT</div><div class="legend-role">CADENCE CONTROLLED</div></div>
      <div class="legend-box"><div class="legend-symbol" id="foodLegendSymbol">●</div><div class="legend-label">TARGET F</div><div class="legend-role" id="foodRole">SEEK // ENERGY</div></div>
      <div class="legend-box"><div class="legend-symbol" id="dangerLegendSymbol">▲</div><div class="legend-label">TARGET X</div><div class="legend-role" id="dangerRole">AVOID // DANGER</div></div>
    </div>
    <div class="arena" id="arena"></div>
  </div>
</div>

<div class="right">
<div class="panel">
  <div class="panel-title"><span>CADENCE DECISION PIPELINE</span><span>LIVE</span></div>
  <div class="brain-flow">
    <div class="flow-title">PERCEPTION VECTOR</div>
    <div class="perception-grid">
      <div class="sense"><div class="sense-name">F ΔX</div><div class="sense-value" id="foodDx">0.00</div></div>
      <div class="sense"><div class="sense-name">F ΔY</div><div class="sense-value" id="foodDy">0.00</div></div>
      <div class="sense"><div class="sense-name">X ΔX</div><div class="sense-value" id="dangerDx">0.00</div></div>
      <div class="sense"><div class="sense-name">X ΔY</div><div class="sense-value" id="dangerDy">0.00</div></div>
      <div class="sense" style="grid-column:1/-1"><div class="sense-name">ENERGY INPUT</div><div class="sense-value" id="energyInput">1.00</div></div>
    </div>
    <div class="flow-arrow">↓</div>
    <div class="brain-row">
      <div class="brain-orb" id="brainOrb">CADENCE<br>PAUSED</div>
      <div class="decision-box"><div class="decision-small">SELECTED ACTION</div><div class="decision-action" id="action">WAIT</div></div>
    </div>
  </div>

  <div class="metrics">
    <div class="metric"><span>IMMEDIATE REWARD</span><strong id="reward">0.00</strong></div>
    <div class="metric"><span>EVENT</span><strong id="event">READY</strong></div>
    <div class="metric"><span>ENERGY</span><strong id="energy">100</strong></div>
    <div class="metric"><span>HEALTH</span><strong id="health">100</strong></div>
    <div class="metric"><span>EPISODE</span><strong id="episode">1</strong></div>
    <div class="metric"><span>STEP</span><strong id="steps">0</strong></div>
    <div class="metric"><span>EPISODE REWARD</span><strong id="episodeReward">0.00</strong></div>
    <div class="metric"><span>CADENCE DOPAMINE</span><strong id="dopamine">0.0000</strong></div>
    <div class="metric"><span>CADENCE TD ERROR</span><strong id="tdError">0.0000</strong></div>
  </div>
</div>

<div class="panel controls">
  <button onclick="toggleRun()" id="runButton">START</button>
  <button class="shift" onclick="triggerShift()" id="shiftButton">⚠ TRIGGER WORLD SHIFT</button>
  <button onclick="resetBrain()">RESET BRAIN</button>
</div>
</div>

<div class="panel bottom">
  <div class="panel-title"><span>LEARNING PERFORMANCE</span><span>WORLD-SHIFT PROOF VIEW</span></div>

  <div class="proof-grid">
    <div class="proof-card"><div class="proof-name">PRE-SHIFT AVG</div><div class="proof-value" id="beforeAvg">—</div></div>
    <div class="proof-card"><div class="proof-name">EARLY POST-SHIFT AVG</div><div class="proof-value" id="earlyAvg">—</div></div>
    <div class="proof-card"><div class="proof-name">RECENT POST-SHIFT AVG</div><div class="proof-value" id="recentAvg">—</div></div>
    <div class="proof-card"><div class="proof-name">POST-SHIFT CHANGE</div><div class="proof-value" id="recovery">—</div></div>
  </div>

  <div class="graph" id="graph"></div>
  <div class="regions"><span>BEFORE SHIFT</span><span>AFTER SHIFT</span></div>

  <div class="footer-note">
    Each bar is one completed episode. Cyan = positive reward; red = negative reward.
    The red marker records the episode where target meanings changed. “Post-shift change”
    compares the most recent three completed post-shift episodes with the first three.
  </div>
</div>

</div>

<script>
function signed(v){return v>0?"+"+v.toFixed(2):v.toFixed(2)}
function metric(v){return v===null||v===undefined?"—":signed(v)}

function drawArena(data){
  const arena=document.getElementById("arena"); arena.innerHTML="";
  for(let y=0;y<data.size;y++){
    for(let x=0;x<data.size;x++){
      const cell=document.createElement("div"); cell.className="cell";
      if(x===data.agent[0]&&y===data.agent[1]) cell.innerHTML='<span class="agent">◆</span>';
      else if(x===data.food[0]&&y===data.food[1]) cell.innerHTML='<span class="'+(data.shifted?"shifted-food":"food")+'">●</span>';
      else if(x===data.danger[0]&&y===data.danger[1]) cell.innerHTML='<span class="'+(data.shifted?"shifted-danger":"danger")+'">▲</span>';
      arena.appendChild(cell);
    }
  }
}

function updateRoles(data){
  const f=document.getElementById("foodLegendSymbol");
  const x=document.getElementById("dangerLegendSymbol");
  if(!data.shifted){
    f.className="legend-symbol green"; x.className="legend-symbol red";
    document.getElementById("foodRole").textContent="SEEK // ENERGY";
    document.getElementById("dangerRole").textContent="AVOID // DANGER";
  }else{
    f.className="legend-symbol red"; x.className="legend-symbol green";
    document.getElementById("foodRole").textContent="AVOID // DANGER";
    document.getElementById("dangerRole").textContent="SEEK // ENERGY";
  }
}

function drawGraph(data){
  const graph=document.getElementById("graph");
  graph.innerHTML='<div class="zero-line"></div>';

  const values=data.reward_history;
  if(!values.length) return;

  let maxValue=1;
  values.forEach(item=>maxValue=Math.max(maxValue,Math.abs(item.reward)));

  values.forEach(item=>{
    const wrap=document.createElement("div"); wrap.className="bar-wrap";
    const bar=document.createElement("div");
    bar.style.height=Math.max(3,Math.abs(item.reward)/maxValue*46)+"%";
    bar.className=item.reward>=0?"bar positive":"bar negative";
    bar.title="Episode "+item.episode+": "+item.reward.toFixed(2);

    const label=document.createElement("div");
    label.className="ep-label";
    label.textContent=item.episode;

    wrap.appendChild(bar); wrap.appendChild(label); graph.appendChild(wrap);
  });

  if(data.shift_episode!==null){
    let firstIndex=values.findIndex(item=>item.episode>=data.shift_episode);
    if(firstIndex<0) firstIndex=values.length;
    const pct=(firstIndex/values.length)*100;

    const marker=document.createElement("div");
    marker.className="shift-marker";
    marker.style.left="calc("+pct+"% - 1px)";
    marker.innerHTML="<span>⚠ WORLD SHIFT · EP "+data.shift_episode+"</span>";
    graph.appendChild(marker);
  }
}

async function updateState(){
  try{
    const response=await fetch("/state");
    const data=await response.json();

    drawArena(data); drawGraph(data); updateRoles(data);

    document.getElementById("worldMode").textContent=data.shifted?"WORLD // SHIFTED":"WORLD // NORMAL";
    document.getElementById("foodDx").textContent=signed(data.perception[0]);
    document.getElementById("foodDy").textContent=signed(data.perception[1]);
    document.getElementById("dangerDx").textContent=signed(data.perception[2]);
    document.getElementById("dangerDy").textContent=signed(data.perception[3]);
    document.getElementById("energyInput").textContent=data.perception[4].toFixed(2);
    document.getElementById("action").textContent=data.action;

    const reward=document.getElementById("reward");
    reward.textContent=signed(data.reward);
    reward.className=data.reward>=0?"green":"red";

    document.getElementById("event").textContent=data.event;
    document.getElementById("energy").textContent=data.energy;
    document.getElementById("health").textContent=data.health;
    document.getElementById("episode").textContent=data.episode;
    document.getElementById("steps").textContent=data.steps;
    document.getElementById("episodeReward").textContent=data.episode_reward.toFixed(2);
    document.getElementById("dopamine").textContent=data.dopamine.toFixed(4);
    document.getElementById("tdError").textContent=data.td_error.toFixed(4);
    document.getElementById("runButton").textContent=data.running?"PAUSE":"START";

    document.getElementById("beforeAvg").textContent=metric(data.averages.before);
    document.getElementById("earlyAvg").textContent=metric(data.averages.early);
    document.getElementById("recentAvg").textContent=metric(data.averages.recent);

    const recovery=document.getElementById("recovery");
    recovery.textContent=metric(data.averages.recovery);
    recovery.className="proof-value "+(data.averages.recovery===null?"":(data.averages.recovery>=0?"green":"red"));

    const orb=document.getElementById("brainOrb");
    if(data.running){orb.classList.add("active");orb.innerHTML="CADENCE<br>SETTLING";}
    else{orb.classList.remove("active");orb.innerHTML="CADENCE<br>PAUSED";}
  }catch(error){console.log(error)}
}

async function toggleRun(){await fetch("/toggle",{method:"POST"});updateState()}

async function triggerShift(){
  const response=await fetch("/shift",{method:"POST"});
  const data=await response.json();

  if(data.shifted){
    const alert=document.getElementById("shiftAlert");
    alert.classList.add("show");
    setTimeout(()=>alert.classList.remove("show"),2600);
  }
  updateState();
}

async function resetBrain(){
  if(!confirm("Reset Cadence and erase its current learned state?")) return;
  await fetch("/reset",{method:"POST"});
  updateState();
}

setInterval(updateState,180);
updateState();
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/state")
def state():
    with LOCK:
        diagnostics = brain.diagnostics()
        senses = perception(world)

        return jsonify({
            "size": world.size,
            "agent": world.agent,
            "food": world.food,
            "danger": world.danger,
            "health": world.health,
            "energy": world.energy,
            "steps": world.steps,
            "episode": episode,
            "action": last_action,
            "reward": last_reward,
            "event": last_event,
            "episode_reward": episode_reward,
            "shifted": shifted,
            "shift_episode": shift_episode,
            "running": running,
            "perception": senses,
            "reward_history": reward_history,
            "averages": averages(),
            "dopamine": diagnostics.get("dopamine", 0.0),
            "td_error": diagnostics.get("td_error", 0.0),
        })


@app.route("/toggle", methods=["POST"])
def toggle():
    global running
    with LOCK:
        running = not running
    return jsonify({"running": running})


@app.route("/shift", methods=["POST"])
def shift():
    global shifted, shift_episode, last_event

    with LOCK:
        if not shifted:
            shifted = True
            shift_episode = episode
            last_event = "WORLD SHIFT"
        else:
            last_event = "WORLD ALREADY SHIFTED"

    return jsonify({
        "shifted": shifted,
        "shift_episode": shift_episode
    })


@app.route("/reset", methods=["POST"])
def reset():
    global brain, world, episode, shifted, shift_episode, running
    global last_action, last_reward, last_event, episode_reward, reward_history

    with LOCK:
        running = False
        brain = SurvivalBrain()
        world = SurvivalWorld()

        episode = 1
        shifted = False
        shift_episode = None

        last_action = "WAIT"
        last_reward = 0.0
        last_event = "BRAIN RESET"
        episode_reward = 0.0
        reward_history = []

    return jsonify({"reset": True})


if __name__ == "__main__":
    worker = threading.Thread(target=simulation_loop, daemon=True)
    worker.start()

    print()
    print("========================================")
    print("   CADENCE // SURVIVAL PROOF DASHBOARD")
    print("========================================")
    print()
    print("Open in browser:")
    print()
    print("http://127.0.0.1:5000")
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )
