pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import "../theme" as Room36Theme

GridView {
    id: root
    objectName: "peopleGrid"
    property var peopleModel: null
    property bool known: true
    property string viewSize: "Medium"
    signal personOpened(string key)
    signal nameRequested(string key)
    signal ignoreRequested(string key, bool ignored)
    Room36Theme.Metrics { id: metrics }

    model: peopleModel
    readonly property int scrollbarGutter: 18
    readonly property real preferredCardWidth: 230
        * (viewSize === "Small" ? 0.8 : viewSize === "Large" ? 1.25 : 1)
    readonly property int columns: Math.max(1, Math.floor((width - scrollbarGutter) / preferredCardWidth))
    cellWidth: Math.max(1, (width - scrollbarGutter) / columns)
    cellHeight: Math.max(180, Math.min(270, cellWidth * 0.88))
    cacheBuffer: cellHeight
    reuseItems: false
    clip: true
    boundsBehavior: Flickable.StopAtBounds
    ScrollBar.vertical: ScrollBar { objectName: "peopleGridScrollBar"; width: 8 }

    delegate: Item {
        id: delegateRoot
        objectName: "peopleGridDelegate"
        required property string key
        required property string name
        required property int faceCount
        required property int photoCount
        required property bool ignored
        required property string previewSource
        property var leaseModel: null
        property int lease: 0
        property bool mounted: false
        width: root.cellWidth
        height: root.cellHeight

        function acquire() {
            if (!mounted) return
            release()
            leaseModel = root.peopleModel
            if (leaseModel) lease = leaseModel.acquirePreview(key, 192)
        }
        function release() {
            if (leaseModel && lease) leaseModel.releasePreview(lease)
            lease = 0
            leaseModel = null
        }
        Timer { id: acquisitionTimer; interval: 0; onTriggered: delegateRoot.acquire() }
        onKeyChanged: if (mounted) acquisitionTimer.restart()
        Component.onCompleted: { mounted = true; acquisitionTimer.start() }
        Component.onDestruction: { mounted = false; release() }
        Connections {
            target: root
            function onPeopleModelChanged() { acquisitionTimer.restart() }
        }
        PeopleCard {
            objectName: "peopleCard"
            anchors.fill: parent
            title: root.known ? delegateRoot.name : ""
            subtitle: delegateRoot.photoCount + (delegateRoot.photoCount === 1 ? " photo" : " photos")
                + (root.known ? " · " + delegateRoot.faceCount
                    + (delegateRoot.faceCount === 1 ? " face" : " faces") : "")
            previewSource: delegateRoot.previewSource
            ignored: delegateRoot.ignored
            showIgnore: !root.known
            onOpened: root.personOpened(delegateRoot.key)
            onNameRequested: root.nameRequested(delegateRoot.key)
            onIgnoreRequested: root.ignoreRequested(delegateRoot.key, !delegateRoot.ignored)
        }
    }
}
